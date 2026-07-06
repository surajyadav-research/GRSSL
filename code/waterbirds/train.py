import os
import re
import csv
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==========================================
# 0. EXPERIMENT CONFIG
# ==========================================

EXPERIMENT_NAME = "waterbirds"  # Used for output directory naming.

# ---------- General ----------
LR           = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.5
BATCH_SIZE   = 64
NUM_WORKERS  = 4
IMAGE_SIZE   = 224

# ---------- Stage 1 SSL ----------
STAGE1_EPOCHS            = 10
SSL_TEMPERATURE          = 0.2
SSL_PROJECTION_DIM       = 128
SSL_PROJECTOR_HIDDEN_DIM = 2048
SSL_START_FROM_IMAGENET  = True
SSL_NUM_VIEWS            = 2

# ---------- Stage 2 Classifier ----------
STAGE2_EPOCHS            = 6
STAGE2_LINEAR_PROBE_ONLY = False
STAGE2_START_FROM_SSL    = True
GROUPDRO_WARMUP_EPOCHS   = 2

# ---------- GroupDRO ----------
NUM_GROUPS         = 4
GROUPDRO_STEP_SIZE = 0.01
EVAL_EVERY_EPOCH   = True

# ---------- Multi-seed ----------
SEEDS = [42, 43, 44]  # Set of random seeds to run. Change as needed.

# ---------- Paths ----------
DECODED_ROOT   = REPO_ROOT / "data" / "waterbirds" / "generated"
ORIGINAL_ROOT  = REPO_ROOT / "data" / "waterbirds" / "waterbirds_original"
EXPERIMENT_DIR = REPO_ROOT / "output" / EXPERIMENT_NAME


# ==========================================
# 1. SEED HELPER
# ==========================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ==========================================
# 2. PER-SEED DIRECTORY SETUP
# ==========================================

def make_seed_dirs(seed: int):
    base = EXPERIMENT_DIR / f"seed_{seed}"

    dirs = {
        "s1_ckpt":     base / "stage1_ssl_pretraining"     / "checkpoints",
        "s1_log":      base / "stage1_ssl_pretraining"     / "logs",
        "s1_analysis": base / "stage1_ssl_pretraining"     / "analysis",
        "s2_ckpt":     base / "stage2_classifier_training" / "checkpoints",
        "s2_log":      base / "stage2_classifier_training" / "logs",
        "s2_analysis": base / "stage2_classifier_training" / "analysis",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return {
        "s1_best_ssl":     dirs["s1_ckpt"]     / "best_ssl_resnet50.pt",
        "s1_final_ssl":    dirs["s1_ckpt"]     / "final_ssl_resnet50.pt",
        "s1_best_enc":     dirs["s1_ckpt"]     / "best_ssl_encoder.pt",
        "s1_final_enc":    dirs["s1_ckpt"]     / "final_ssl_encoder.pt",
        "s1_train_log":    dirs["s1_log"]      / "ssl_train_log.csv",
        "s2_best_model":   dirs["s2_ckpt"]     / "best_classifier.pt",
        "s2_final_model":  dirs["s2_ckpt"]     / "final_classifier.pt",
        "s2_train_log":    dirs["s2_log"]      / "train_log.csv",
        "s2_eval_log":     dirs["s2_log"]      / "eval_log.csv",
        "s2_test_results": dirs["s2_analysis"] / "test_results.csv",
    }


# ==========================================
# 3. ARCHITECTURE
# ==========================================

def create_resnet50_backbone(pretrained=False):
    weights  = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim


class SSLResNet50(nn.Module):
    def __init__(self, projection_dim=128, projector_hidden_dim=2048,
                 pretrained_backbone=False):
        super().__init__()
        self.encoder, feat_dim = create_resnet50_backbone(pretrained=pretrained_backbone)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x):
        features = self.encoder(x)
        return features, self.projector(features)


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5, pretrained_backbone=False):
        super().__init__()
        self.encoder, feat_dim = create_resnet50_backbone(pretrained=pretrained_backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def get_param_groups(self, base_lr):
        layer4_ids   = {id(p) for p in self.encoder.layer4.parameters()}
        early_params = [p for p in self.encoder.parameters() if id(p) not in layer4_ids]
        return [
            {"params": early_params,                           "lr": base_lr * 0.1},
            {"params": list(self.encoder.layer4.parameters()), "lr": base_lr * 0.5},
            {"params": list(self.classifier.parameters()),     "lr": base_lr},
        ]


# ==========================================
# 4. METADATA HELPERS
# ==========================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def append_row_to_csv(csv_path, header, row):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def group_id_from_label_place(label, place):
    return int(label) * 2 + int(place)


def load_sidecar_csv(csv_path: Path):
    data = {}
    if not csv_path.exists():
        return data
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                data[row[0]] = row[1]
    return data


def parse_int_safely(v):
    if v is None:
        return None
    v = str(v).strip()
    return None if v == "" else int(float(v))


def detect_column(row_keys, candidates):
    lowered = {k.lower(): k for k in row_keys}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def load_metadata_csv(path: Path):
    meta = {}
    if not path.exists():
        return meta
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return meta
        fn = detect_column(
            reader.fieldnames,
            ["filename", "file_name", "img_filename", "image_filename", "path"]
        )
        lc = detect_column(reader.fieldnames, ["y", "label", "meta_label", "target"])
        pc = detect_column(reader.fieldnames, ["place", "meta_place", "background", "bg"])

        if None in (fn, lc, pc):
            raise ValueError(f"Missing required columns in {path}. Found: {reader.fieldnames}")

        for row in reader:
            raw   = str(row[fn]).strip()
            label = parse_int_safely(row[lc])
            place = parse_int_safely(row[pc])

            if not raw or label is None or place is None:
                continue

            p = Path(raw)
            keys = {
                raw,
                p.name,
                p.stem,
                str(p),
                str(p).replace("\\", "/"),
            }

            raw_norm = raw.replace("\\", "/")
            if "images/" in raw_norm:
                rel_from_images = raw_norm.split("images/", 1)[-1]
                keys.add(rel_from_images)
                keys.add(Path(rel_from_images).name)
                keys.add(Path(rel_from_images).stem)

            for key in keys:
                meta[key] = {"label": label, "place": place}

    return meta


def resolve_images_dir_and_metadata(split_dir: Path):
    split_dir  = Path(split_dir)
    images_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    meta_csv   = split_dir / "metadata.csv"
    return images_dir, (load_metadata_csv(meta_csv) if meta_csv.exists() else {})


def get_label_place_for_file(file_path: Path, metadata_map: dict, images_dir: Path = None):
    sidecar = file_path.with_suffix(".csv")
    if sidecar.exists():
        sc = load_sidecar_csv(sidecar)
        if "meta_label" in sc and "meta_place" in sc:
            return int(sc["meta_label"]), int(sc["meta_place"])

    candidates = [
        file_path.name,
        file_path.stem,
        str(file_path),
        str(file_path).replace("\\", "/"),
    ]

    if images_dir is not None:
        try:
            rel = file_path.relative_to(images_dir)
            candidates.extend([
                str(rel),
                str(rel).replace("\\", "/"),
                rel.name,
                rel.stem,
            ])
        except ValueError:
            pass

    for key in candidates:
        if key in metadata_map:
            return metadata_map[key]["label"], metadata_map[key]["place"]

    return None, None


def get_group_key(stem: str) -> str:
    m = re.match(r"^(\d+)_(orig|var\d+)$", stem)
    return m.group(1) if m else stem


# ==========================================
# 5. DATASETS
# ==========================================

class MultiViewSSLDataset(Dataset):
    """
    Groups 000000_orig / 000000_var0..var3 by their numeric prefix.
    Each __getitem__ samples num_views=2 distinct variants of the same original
    so that InfoNCE learns background invariance from pre-generated diversity
    instead of relying solely on augmentation.
    """
    def __init__(self, split_dir, transform=None, num_views=2):
        self.transform = transform
        self.num_views = num_views

        images_dir, meta = resolve_images_dir_and_metadata(Path(split_dir))

        groups     = defaultdict(list)
        group_meta = {}

        for fp in sorted(images_dir.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS:
                continue
            if not re.match(r"^\d+_(orig|var\d+)$", fp.stem):
                continue
            label, place = get_label_place_for_file(fp, meta, images_dir)
            if label is None:
                continue
            key = get_group_key(fp.stem)
            groups[key].append(fp)
            group_meta[key] = (label, place)

        self.groups    = []
        self.group_ids = []
        for key, files in groups.items():
            if len(files) < 2:
                continue
            label, place = group_meta[key]
            self.groups.append(files)
            self.group_ids.append(group_id_from_label_place(label, place))

        print(f"  MultiViewSSLDataset: {len(self.groups)} groups, "
              f"avg {sum(len(g) for g in self.groups)/max(len(self.groups),1):.1f} views/group")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        chosen = random.sample(self.groups[idx], k=min(self.num_views, len(self.groups[idx])))
        views  = []
        for path in chosen:
            img = Image.open(path).convert("RGB")
            views.append(self.transform(img) if self.transform else transforms.ToTensor()(img))
        return views

    def get_sample_weights(self):
        counts = Counter(self.group_ids)
        return [1.0 / counts[g] for g in self.group_ids], counts


class GroupBalancedClassificationDataset(Dataset):
    """All 5 variants (orig + var0..var3) used as independent training samples."""
    def __init__(self, split_dir, transform=None):
        self.transform = transform
        images_dir, self.meta = resolve_images_dir_and_metadata(Path(split_dir))

        self.files     = []
        self.group_ids = []

        for fp in sorted(images_dir.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS:
                continue
            label, place = get_label_place_for_file(fp, self.meta, images_dir)
            if label is None:
                continue
            self.files.append(fp)
            self.group_ids.append(group_id_from_label_place(label, place))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp           = self.files[idx]
        label, place = get_label_place_for_file(fp, self.meta, self.images_dir if hasattr(self, "images_dir") else None)
        if label is None:
            raise ValueError(f"Missing metadata: {fp}")
        image = Image.open(fp).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, place, group_id_from_label_place(label, place), fp.stem

    def get_sample_weights(self):
        counts = Counter(self.group_ids)
        return [1.0 / counts[g] for g in self.group_ids], counts


class EvalClassificationDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.transform = transform
        self.images_dir, self.meta = resolve_images_dir_and_metadata(Path(split_dir))
        self.files = []

        for fp in sorted(self.images_dir.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS:
                continue
            label, place = get_label_place_for_file(fp, self.meta, self.images_dir)
            if label is not None and place is not None:
                self.files.append(fp)

        if len(self.files) == 0:
            raise ValueError(f"No valid evaluation images found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp           = self.files[idx]
        label, place = get_label_place_for_file(fp, self.meta, self.images_dir)
        if label is None or place is None:
            raise ValueError(f"Missing metadata: {fp}")
        image = Image.open(fp).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, place, fp.stem


# ==========================================
# 6. LOSS / LOADER HELPERS
# ==========================================

def nt_xent_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B  = z1.size(0)
    if B < 2:
        raise ValueError("Batch size must be >= 2 for InfoNCE. Use drop_last=True.")
    rep  = torch.cat([z1, z2], dim=0)
    sim  = torch.matmul(rep, rep.T) / temperature
    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim  = sim.masked_fill(mask, -1e9)
    pos  = torch.arange(B, device=z1.device)
    pos  = torch.cat([pos + B, pos], dim=0)
    return F.cross_entropy(sim, pos)


def ssl_collate_fn(batch):
    """Collate list-of-views batches from MultiViewSSLDataset."""
    num_views = len(batch[0])
    return [torch.stack([s[i] for s in batch]) for i in range(num_views)]


def build_balanced_loader(dataset, batch_size, num_workers, drop_last=False,
                           collate_fn=None):
    weights, counts = dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader, counts


# ==========================================
# 7. GROUPDRO WITH ERM WARMUP
# ==========================================

class GroupDROCriterion(nn.Module):
    """
    GroupDRO with ERM warmup.
    For the first `warmup_epochs` epochs uses uniform group weights (plain ERM).
    After warmup, switches to the adversarial q-update so that the classifier
    head is already stable before worst-group emphasis kicks in.
    """
    def __init__(self, num_groups=4, step_size=0.01, warmup_epochs=2, device="cpu"):
        super().__init__()
        self.num_groups    = num_groups
        self.step_size     = step_size
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.register_buffer("q", torch.ones(num_groups, device=device) / num_groups)
        self.per_sample_ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, group_ids):
        per_sample = self.per_sample_ce(logits, targets)

        group_losses = torch.zeros(self.num_groups, device=logits.device)
        group_counts = torch.zeros(self.num_groups, device=logits.device)
        for g in range(self.num_groups):
            mask = (group_ids == g)
            if mask.any():
                group_losses[g] = per_sample[mask].mean()
                group_counts[g] = mask.sum()

        observed = group_counts > 0

        if self.current_epoch < self.warmup_epochs:
            robust_loss = group_losses[observed].mean()
        else:
            with torch.no_grad():
                self.q[observed] *= torch.exp(self.step_size * group_losses[observed])
                self.q /= self.q.sum()
            robust_loss = (self.q[observed] * group_losses[observed]).sum()

        return robust_loss, group_losses.detach(), group_counts.detach(), self.q.detach().clone()


# ==========================================
# 8. EVALUATION
# ==========================================

def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    groups = {
        (0, 0): {"name": "Landbird on Land",   "correct": 0, "total": 0},
        (0, 1): {"name": "Landbird on Water",  "correct": 0, "total": 0},
        (1, 0): {"name": "Waterbird on Land",  "correct": 0, "total": 0},
        (1, 1): {"name": "Waterbird on Water", "correct": 0, "total": 0},
    }
    with torch.no_grad():
        for images, labels, places, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            places = places.to(device, non_blocking=True)
            logits = model(images)
            preds  = torch.argmax(logits, dim=1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
            for i in range(labels.size(0)):
                lbl, plc = labels[i].item(), places[i].item()
                if (lbl, plc) in groups:
                    groups[(lbl, plc)]["total"] += 1
                    if preds[i].item() == lbl:
                        groups[(lbl, plc)]["correct"] += 1

    overall_acc = 100 * correct / total if total > 0 else 0.0
    group_accs  = {
        k: (100 * s["correct"] / s["total"] if s["total"] > 0 else float("nan"))
        for k, s in groups.items()
    }
    valid = [a for a in group_accs.values() if not math.isnan(a)]
    worst = min(valid) if valid else float("nan")
    return overall_acc, worst, group_accs, groups, correct, total


def print_eval_results(split_name, overall_acc, worst_group_acc, group_accs,
                        groups, correct, total):
    print(f"\n{'='*15} {split_name.upper()} {'='*15}")
    print(f"Overall:      {overall_acc:.2f}%  ({correct}/{total})")
    print("-" * 42)
    for k, s in groups.items():
        a = group_accs[k]
        line = f"{a:>6.2f}%  ({s['correct']}/{s['total']})" \
               if not math.isnan(a) else "N/A"
        print(f"  {s['name']:<22}: {line}")
    print("-" * 42)
    print(f"Worst-group:  {worst_group_acc:.2f}")
    print("=" * 42)


def save_test_results_csv(path, split, overall, worst, group_accs, groups, correct, total):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "overall_acc", "worst_group_acc", "correct", "total"])
        w.writerow([split, f"{overall:.2f}", f"{worst:.2f}" if not math.isnan(worst) else "nan", correct, total])
        w.writerow([])
        w.writerow(["group_key", "group_name", "accuracy", "correct", "total"])
        for k, s in groups.items():
            a = group_accs[k]
            w.writerow([str(k), s["name"],
                        f"{a:.2f}" if not math.isnan(a) else "nan",
                        s["correct"], s["total"]])


# ==========================================
# 9. TRAINING FUNCTIONS
# ==========================================

def train_ssl_model(model, train_loader, device, epochs, lr, weight_decay,
                    temperature, paths):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")

    print("\n  --- Stage 1: SSL Pretraining ---")
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for views in train_loader:
            v1 = views[0].to(device, non_blocking=True)
            v2 = views[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss  = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  Epoch [{epoch+1}/{epochs}] SSL Loss: {avg:.4f}")
        append_row_to_csv(paths["s1_train_log"], ["epoch", "ssl_loss"],
                          [epoch + 1, f"{avg:.6f}"])

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(),         paths["s1_best_ssl"])
            torch.save(model.encoder.state_dict(), paths["s1_best_enc"])
            print(f"  -> New best SSL loss: {best_loss:.4f}. Saved.")

    torch.save(model.state_dict(),         paths["s1_final_ssl"])
    torch.save(model.encoder.state_dict(), paths["s1_final_enc"])
    return best_loss


def train_classifier_model(model, train_loader, val_loader, device, epochs,
                             lr, weight_decay, paths, linear_probe_only=False,
                             num_groups=4, groupdro_step_size=0.01,
                             groupdro_warmup_epochs=2):
    if linear_probe_only:
        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay)
        print("  Stage 2: Linear probe only (encoder frozen)")
    else:
        optimizer = torch.optim.AdamW(
            model.get_param_groups(lr), weight_decay=weight_decay)
        print("  Stage 2: Layer-wise LR fine-tuning")

    groupdro = GroupDROCriterion(
        num_groups=num_groups,
        step_size=groupdro_step_size,
        warmup_epochs=groupdro_warmup_epochs,
        device=device,
    )

    best_wga = -1.0

    print(f"\n  --- Stage 2: Classifier Training (warmup={groupdro_warmup_epochs} epochs) ---")
    for epoch in range(epochs):
        model.train()
        groupdro.current_epoch = epoch

        total_loss, correct, total = 0.0, 0, 0
        gl_sum = torch.zeros(num_groups, device=device)
        gc_sum = torch.zeros(num_groups, device=device)

        for images, labels, places, group_ids, _ in train_loader:
            images    = images.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)
            group_ids = group_ids.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            robust_loss, batch_gl, batch_gc, q = groupdro(outputs, labels, group_ids)
            robust_loss.backward()
            optimizer.step()

            total_loss += robust_loss.item()
            preds       = torch.argmax(outputs, dim=1)
            total      += labels.size(0)
            correct    += (preds == labels).sum().item()

            obs = batch_gc > 0
            gl_sum[obs] += batch_gl[obs] * batch_gc[obs]
            gc_sum[obs] += batch_gc[obs]

        avg_loss  = total_loss / max(len(train_loader), 1)
        train_acc = 100 * correct / total if total > 0 else 0.0
        mode_str  = "ERM-warmup" if epoch < groupdro_warmup_epochs else "GroupDRO"

        epoch_gl = torch.full((num_groups,), float("nan"), device=device)
        present  = gc_sum > 0
        epoch_gl[present] = gl_sum[present] / gc_sum[present]

        print(f"  Epoch [{epoch+1}/{epochs}] [{mode_str}] "
              f"Loss:{avg_loss:.4f}  Acc:{train_acc:.2f}%  "
              f"q:{[round(x, 4) for x in q.tolist()]}")

        append_row_to_csv(
            paths["s2_train_log"],
            ["epoch", "mode", "loss", "train_acc",
             "gl_0", "gl_1", "gl_2", "gl_3",
             "q_0",  "q_1",  "q_2",  "q_3"],
            [
                epoch + 1, mode_str, f"{avg_loss:.6f}", f"{train_acc:.2f}",
                f"{epoch_gl[0].item():.6f}" if present[0] else "nan",
                f"{epoch_gl[1].item():.6f}" if present[1] else "nan",
                f"{epoch_gl[2].item():.6f}" if present[2] else "nan",
                f"{epoch_gl[3].item():.6f}" if present[3] else "nan",
                f"{q[0].item():.6f}", f"{q[1].item():.6f}",
                f"{q[2].item():.6f}", f"{q[3].item():.6f}",
            ],
        )

        if EVAL_EVERY_EPOCH and val_loader is not None:
            overall, worst, group_accs, g_stats, c, t = evaluate_model(
                model, val_loader, device)
            print(f"  -> Val Overall: {overall:.2f}%  Worst-Group: {worst:.2f}%")

            if worst > best_wga:
                best_wga = worst
                torch.save(model.state_dict(), paths["s2_best_model"])
                print(f"  -> New best WGA: {best_wga:.2f}%. Saved.")

            append_row_to_csv(
                paths["s2_eval_log"],
                ["epoch", "split", "overall_acc", "worst_group_acc",
                 "acc_ll", "acc_lw", "acc_wl", "acc_ww"],
                [
                    epoch + 1, "validation",
                    f"{overall:.2f}", f"{worst:.2f}",
                    f"{group_accs.get((0,0), float('nan')):.2f}",
                    f"{group_accs.get((0,1), float('nan')):.2f}",
                    f"{group_accs.get((1,0), float('nan')):.2f}",
                    f"{group_accs.get((1,1), float('nan')):.2f}",
                ],
            )

    torch.save(model.state_dict(), paths["s2_final_model"])
    print(f"\n  Final model saved: {paths['s2_final_model']}")

    if val_loader is None and not paths["s2_best_model"].exists():
        torch.save(model.state_dict(), paths["s2_best_model"])
        best_wga = float("nan")

    return best_wga


# ==========================================
# 10. SINGLE-SEED RUN
# ==========================================

def run_single_seed(seed, device, ssl_transform, classifier_train_transform,
                    eval_transform):
    print(f"\n{'#'*52}")
    print(f"  SEED {seed}")
    print(f"{'#'*52}")

    set_seed(seed)
    paths = make_seed_dirs(seed)

    TRAIN_DIR = DECODED_ROOT / "train"
    VAL_DIR   = DECODED_ROOT / "validation"

    # Fixed test path: use original images test set
    TEST_DIR  = ORIGINAL_ROOT / "test"

    # --- Stage 1 data ---
    ssl_ds = MultiViewSSLDataset(
        TRAIN_DIR, transform=ssl_transform, num_views=SSL_NUM_VIEWS)
    ssl_loader, ssl_counts = build_balanced_loader(
        ssl_ds, BATCH_SIZE, NUM_WORKERS,
        drop_last=True, collate_fn=ssl_collate_fn)
    print(f"  SSL groups: {len(ssl_ds)}  group counts: {ssl_counts}")

    # --- Stage 1 model ---
    s1_model = SSLResNet50(
        projection_dim=SSL_PROJECTION_DIM,
        projector_hidden_dim=SSL_PROJECTOR_HIDDEN_DIM,
        pretrained_backbone=SSL_START_FROM_IMAGENET,
    ).to(device)

    train_ssl_model(s1_model, ssl_loader, device, STAGE1_EPOCHS,
                    LR, WEIGHT_DECAY, SSL_TEMPERATURE, paths)

    # --- Stage 2 data ---
    s2_ds = GroupBalancedClassificationDataset(
        TRAIN_DIR, transform=classifier_train_transform)
    print(f"  Stage 2 train samples: {len(s2_ds)}")
    print(f"  Stage 2 group counts : {Counter(s2_ds.group_ids)}")

    s2_loader, _ = build_balanced_loader(
        s2_ds, BATCH_SIZE, NUM_WORKERS, drop_last=False)

    val_loader = None
    if VAL_DIR.exists():
        val_ds = EvalClassificationDataset(VAL_DIR, transform=eval_transform)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            print(f"  Val samples: {len(val_ds)}")

    # --- Stage 2 model ---
    s2_model = ResNet50Classifier(
        num_classes=2, dropout=DROPOUT, pretrained_backbone=False).to(device)

    if STAGE2_START_FROM_SSL:
        enc_path = (paths["s1_best_enc"] if paths["s1_best_enc"].exists()
                    else paths["s1_final_enc"])
        if enc_path.exists():
            s2_model.encoder.load_state_dict(
                torch.load(enc_path, map_location=device))
            print(f"  Loaded SSL encoder: {enc_path}")
        else:
            print("  Warning: No SSL encoder checkpoint found.")

    train_classifier_model(
        s2_model, s2_loader, val_loader, device,
        STAGE2_EPOCHS, LR, WEIGHT_DECAY, paths,
        linear_probe_only=STAGE2_LINEAR_PROBE_ONLY,
        num_groups=NUM_GROUPS,
        groupdro_step_size=GROUPDRO_STEP_SIZE,
        groupdro_warmup_epochs=GROUPDRO_WARMUP_EPOCHS,
    )

    # --- Test evaluation on best checkpoint ---
    test_overall, test_worst = float("nan"), float("nan")
    test_group_accs, g_stats = {}, {}

    if TEST_DIR.exists():
        print(f"  Using test directory: {TEST_DIR}")
        test_ds = EvalClassificationDataset(TEST_DIR, transform=eval_transform)
        print(f"  Test samples: {len(test_ds)}")

        if len(test_ds) > 0:
            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available()
            )

            load_path = None
            if paths["s2_best_model"].exists():
                load_path = paths["s2_best_model"]
            elif paths["s2_final_model"].exists():
                load_path = paths["s2_final_model"]

            if load_path is not None:
                state = torch.load(load_path, map_location=device)
                s2_model.load_state_dict(state)
                print(f"  Loaded checkpoint for test: {load_path}")
            else:
                print("  Warning: No classifier checkpoint found for test. Evaluating current in-memory model.")

            test_overall, test_worst, test_group_accs, g_stats, c, t = \
                evaluate_model(s2_model, test_loader, device)

            print_eval_results(f"seed {seed} test", test_overall, test_worst,
                               test_group_accs, g_stats, c, t)
            save_test_results_csv(
                paths["s2_test_results"], f"seed_{seed}_test",
                test_overall, test_worst, test_group_accs, g_stats, c, t)
    else:
        print(f"  Warning: Test directory does not exist: {TEST_DIR}")

    return {
        "seed":         seed,
        "test_overall": test_overall,
        "test_worst":   test_worst,
        "group_accs":   test_group_accs,
    }


# ==========================================
# 11. AGGREGATE RESULTS  (mean ± std)
# ==========================================

def fmt_mean_std(values):
    """Format a list of floats as 'XX.X ± Y.Y'."""
    if not values:
        return "N/A"
    mean = float(np.mean(values))
    std  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.1f} ± {std:.1f}"


def aggregate_and_save(results: list):
    overall_vals = [r["test_overall"] for r in results
                    if not math.isnan(r["test_overall"])]
    worst_vals   = [r["test_worst"]   for r in results
                    if not math.isnan(r["test_worst"])]

    group_keys  = [(0,0), (0,1), (1,0), (1,1)]
    group_names = {
        (0,0): "Landbird on Land",
        (0,1): "Landbird on Water",
        (1,0): "Waterbird on Land",
        (1,1): "Waterbird on Water",
    }
    group_vals = {k: [] for k in group_keys}
    for r in results:
        for k in group_keys:
            v = r["group_accs"].get(k, float("nan"))
            if not math.isnan(v):
                group_vals[k].append(v)

    seed_list = [r["seed"] for r in results]

    print(f"\n{'='*52}")
    print(f"  MULTI-SEED SUMMARY  (seeds: {seed_list})")
    print(f"{'='*52}")
    print(f"  Overall accuracy  : {fmt_mean_std(overall_vals)}")
    print(f"  Worst-group acc   : {fmt_mean_std(worst_vals)}")
    print(f"  ---")
    for k in group_keys:
        print(f"  {group_names[k]:<24}: {fmt_mean_std(group_vals[k])}")
    print(f"{'='*52}\n")

    summary_path = EXPERIMENT_DIR / "summary_results.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["metric", "mean", "std", "formatted"] +
            [f"seed_{r['seed']}" for r in results]
        )

        def make_row(name, values, per_seed_vals):
            mean = float(np.mean(values)) if values else float("nan")
            std  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            return [name,
                    f"{mean:.2f}" if not math.isnan(mean) else "nan",
                    f"{std:.2f}"  if not math.isnan(std)  else "nan",
                    fmt_mean_std(values),
                    *per_seed_vals]

        writer.writerow(make_row(
            "overall_accuracy",
            overall_vals,
            [f"{r['test_overall']:.2f}" if not math.isnan(r["test_overall"]) else "nan" for r in results],
        ))
        writer.writerow(make_row(
            "worst_group_accuracy",
            worst_vals,
            [f"{r['test_worst']:.2f}" if not math.isnan(r["test_worst"]) else "nan" for r in results],
        ))
        for k in group_keys:
            writer.writerow(make_row(
                f"group_{group_names[k].replace(' ', '_')}",
                group_vals[k],
                [f"{r['group_accs'].get(k, float('nan')):.2f}" for r in results],
            ))

    print(f"  Summary CSV saved: {summary_path}")
    return summary_path


# ==========================================
# 12. MAIN
# ==========================================

def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    print(f"Running seeds: {SEEDS}")

    ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.4),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    classifier_train_transform = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE * 1.15), int(IMAGE_SIZE * 1.15))),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_results = []
    for seed in SEEDS:
        result = run_single_seed(
            seed, device,
            ssl_transform, classifier_train_transform, eval_transform,
        )
        all_results.append(result)

    aggregate_and_save(all_results)

    print(f"All artifacts saved under: {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()