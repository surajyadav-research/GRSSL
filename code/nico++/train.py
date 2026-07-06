import os
import csv
import math
import random
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==========================================
# 0. EXPERIMENT CONFIG
# ==========================================

EXPERIMENT_NAME = "nico++"

# ---------- General ----------
BATCH_SIZE   = 64
NUM_WORKERS  = 4
IMAGE_SIZE   = 224
NUM_CLASSES  = 2

# Train groups:
# fox-grass, fox-outdoor, fox-rock, fox-water,
# wolf-grass, wolf-outdoor, wolf-rock, wolf-water
NUM_GROUPS   = 8

SEEDS        = [42, 43, 44, 45, 46, 47, 48]

# ---------- Classes ----------
CLASS_TO_ID = {
    "fox": 0,
    "wolf": 1,
}

ID_TO_CLASS = {
    0: "Fox",
    1: "Wolf",
}

# ---------- Generated train contexts ----------
TRAIN_CONTEXT_TO_ID = {
    "grass": 0,
    "outdoor": 1,
    "rock": 2,
    "water": 3,
}

ID_TO_TRAIN_CONTEXT = {
    0: "Grass",
    1: "Outdoor",
    2: "Rock",
    3: "Water",
}

# ---------- Real test contexts ----------
# Used only for val/test reporting.
TEST_CONTEXT_TO_ID = {
    "autumn": 0,
    "dim": 1,
}

ID_TO_TEST_CONTEXT = {
    0: "Autumn",
    1: "Dim",
}

# ---------- Stage 1 SSL ----------
STAGE1_EPOCHS            = 10
SSL_TEMPERATURE          = 0.2
SSL_PROJECTION_DIM       = 128
SSL_START_FROM_IMAGENET  = True
SSL_NUM_VIEWS            = 2

SSL_LR           = 1e-4
SSL_WEIGHT_DECAY = 1e-4

# User requested 512 dim for SSL.
# In your original code, SSL_HIDDEN_DIM was logged as "ssl_dim",
# so this is set to 512.
SSL_HIDDEN_DIM   = 512

# ---------- Stage 2 Classifier ----------
STAGE2_EPOCHS            = 6
STAGE2_START_FROM_SSL    = True
GROUPDRO_WARMUP_EPOCHS   = 2
GROUPDRO_STEP_SIZE       = 0.01

# Fixed Classifier Hyperparameters
CLS_LR                = 1e-4
CLS_WEIGHT_DECAY      = 1e-4
CLS_DROPOUT           = 0.5
CLS_LINEAR_PROBE_ONLY = False

# ---------- Data Roots ----------
TRAIN_ROOT = REPO_ROOT / "data" / "nico++" / "generated" / "train"
TEST_ROOT  = REPO_ROOT / "data" / "nico++" / "nico_original" / "test"

# Setup Master Directory
EXPERIMENT_DIR = REPO_ROOT / "output" / EXPERIMENT_NAME
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_SUMMARY_CSV = EXPERIMENT_DIR / "master_summary.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ==========================================
# 1. ARCHITECTURE
# ==========================================

def create_resnet50_backbone(pretrained=False):
    weights  = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim


class SSLResNet50(nn.Module):
    def __init__(self, projection_dim=128, projector_hidden_dim=512, pretrained_backbone=False):
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
# 2. HELPER UTILITIES
# ==========================================

def append_row_to_csv(csv_path, header, row):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def save_json(json_path, obj):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keeping your original deterministic setting.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ==========================================
# 3. METADATA HELPERS
# ==========================================

def load_sidecar_csv(csv_path: Path):
    data = {}
    if not csv_path.exists():
        return data

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) >= 2 and row[0].strip():
                data[row[0].strip()] = row[1].strip()

    return data


def class_id_from_text(text: str):
    t = str(text).strip().lower()
    return CLASS_TO_ID.get(t, None)


def infer_train_context_from_text(text: str):
    if not text:
        return None

    t = str(text).strip().lower()

    if "grass" in t or "grassy" in t or "meadow" in t:
        return TRAIN_CONTEXT_TO_ID["grass"]

    if (
        "outdoor" in t
        or "highway" in t
        or "asphalt" in t
        or "road" in t
        or "pavement" in t
        or "urban" in t
    ):
        return TRAIN_CONTEXT_TO_ID["outdoor"]

    if (
        "rock" in t
        or "boulder" in t
        or "stone" in t
        or "formation" in t
    ):
        return TRAIN_CONTEXT_TO_ID["rock"]

    if (
        "water" in t
        or "pond" in t
        or "ripples" in t
        or "river" in t
        or "lake" in t
    ):
        return TRAIN_CONTEXT_TO_ID["water"]

    return None


def infer_test_context_from_path(path: Path):
    parts = [p.lower() for p in path.parts]

    if "autumn" in parts:
        return TEST_CONTEXT_TO_ID["autumn"]

    if "dim" in parts:
        return TEST_CONTEXT_TO_ID["dim"]

    return None


def group_id_from_label_place(label, place):
    """
    Used for training groups.
    4 train contexts:
    0: grass
    1: outdoor
    2: rock
    3: water

    group_id = class_id * 4 + context_id
    """
    return int(label) * 4 + int(place)


def test_group_id_from_label_context(label, context):
    """
    Used only for test/val reporting.
    2 real test contexts:
    0: autumn
    1: dim

    group_id = class_id * 2 + context_id
    """
    return int(label) * 2 + int(context)


def parse_train_context_from_sidecar(sidecar):
    image_type = sidecar.get("type", "").strip().lower()

    if image_type == "generated_image":
        context = infer_train_context_from_text(sidecar.get("prompt", ""))

        if context is not None:
            return context

        variation_index = sidecar.get("variation_index", "")
        try:
            variation_index = int(variation_index)
            if variation_index in [0, 1, 2, 3]:
                return variation_index
        except Exception:
            pass

    source_context = sidecar.get("source_context", "")
    context = infer_train_context_from_text(source_context)

    if context is not None:
        return context

    source_path = sidecar.get("source_path", "")
    context = infer_train_context_from_text(source_path)

    return context


def parse_source_train_context_from_sidecar(sidecar):
    """
    Context of the original NICO source image.

    SSL groups multiple generated variants from the same source image together.
    For that use case, group metadata must come from source_context/source_path,
    not from each generated prompt, otherwise later variants can overwrite the
    group context.
    """
    source_context = sidecar.get("source_context", "")
    context = infer_train_context_from_text(source_context)

    if context is not None:
        return context

    source_path = sidecar.get("source_path", "")
    return infer_train_context_from_text(source_path)


def get_source_key_from_sidecar(sidecar, csv_path: Path):
    source_path = sidecar.get("source_path", "").strip()
    if source_path:
        return source_path

    source_file = sidecar.get("source_file", "").strip()
    class_name = sidecar.get("class_name", "").strip()
    source_context = sidecar.get("source_context", "").strip()

    if source_file:
        return f"{class_name}::{source_context}::{source_file}"

    return csv_path.stem


def get_meta_for_test_file(file_path: Path):
    path_parts = [p.lower() for p in file_path.parts]

    label = None
    if "fox" in path_parts:
        label = CLASS_TO_ID["fox"]
    elif "wolf" in path_parts:
        label = CLASS_TO_ID["wolf"]

    place = infer_test_context_from_path(file_path)

    if label is None or place is None:
        return None, None

    return label, place


# ==========================================
# 4. DATASETS
# ==========================================

class NicoMultiViewSSLDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_views=2):
        self.transform = transform
        self.num_views = num_views
        root = Path(root_dir)

        groups = defaultdict(list)
        group_meta = {}

        for csv_path in sorted(root.glob("*.csv")):
            sidecar = load_sidecar_csv(csv_path)
            if not sidecar:
                continue

            img_name = sidecar.get("output_image")
            if not img_name:
                continue

            img_path = root / img_name
            if not img_path.exists():
                continue

            group_key = get_source_key_from_sidecar(sidecar, csv_path)
            if not group_key:
                continue

            label = class_id_from_text(sidecar.get("class_name", ""))

            place = parse_source_train_context_from_sidecar(sidecar)

            if label is not None and place is not None:
                groups[group_key].append(img_path)
                group_meta.setdefault(group_key, (label, place))

        self.groups, self.group_ids = [], []

        for key, files in groups.items():
            if len(files) < 2:
                continue

            label, place = group_meta[key]
            self.groups.append(files)
            self.group_ids.append(group_id_from_label_place(label, place))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        files = self.groups[idx]

        if len(files) >= self.num_views:
            chosen = random.sample(files, k=self.num_views)
        else:
            chosen = random.choices(files, k=self.num_views)

        views = []
        for path in chosen:
            img = Image.open(path).convert("RGB")
            views.append(self.transform(img) if self.transform else transforms.ToTensor()(img))

        return views

    def get_sample_weights(self):
        counts = Counter(self.group_ids)
        return [1.0 / counts[g] for g in self.group_ids], counts


class NicoGeneratedClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        root = Path(root_dir)

        self.files, self.group_ids = [], []

        for csv_path in sorted(root.glob("*.csv")):
            sidecar = load_sidecar_csv(csv_path)
            if not sidecar:
                continue

            img_name = sidecar.get("output_image")
            if not img_name:
                continue

            img_path = root / img_name
            if not img_path.exists():
                continue

            label = class_id_from_text(sidecar.get("class_name", ""))
            place = parse_train_context_from_sidecar(sidecar)

            if label is not None and place is not None:
                self.files.append(img_path)
                self.group_ids.append(group_id_from_label_place(label, place))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        sidecar = load_sidecar_csv(fp.with_suffix(".csv"))

        label = class_id_from_text(sidecar.get("class_name", ""))
        place = parse_train_context_from_sidecar(sidecar)

        if label is None:
            raise RuntimeError(f"Could not parse class_name for: {fp}")

        if place is None:
            raise RuntimeError(f"Could not parse train context for: {fp}")

        group_id = group_id_from_label_place(label, place)

        image = Image.open(fp).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, place, group_id, fp.stem


class DatasetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label, place, group_id, stem = self.subset[idx]

        # In your original code, transform is applied here.
        # For this to work correctly, underlying dataset should return PIL image.
        # But your original NicoGeneratedClassificationDataset already applied transform.
        # So this wrapper is mainly useful for test dataset below.
        if self.transform and isinstance(img, Image.Image):
            img = self.transform(img)

        return img, label, place, group_id, stem

    def __len__(self):
        return len(self.subset)


class NicoNestedTestDataset(Dataset):
    """
    Expected structure:

    TEST_ROOT/
      autumn/
        fox/
        wolf/
      dim/
        fox/
        wolf/
    """

    def __init__(self, test_root, transform=None):
        self.transform = transform
        self.files = []

        test_root = Path(test_root)

        for context_name in TEST_CONTEXT_TO_ID.keys():
            for class_name in CLASS_TO_ID.keys():
                folder = test_root / context_name / class_name

                if not folder.exists():
                    print(f"WARNING: missing folder: {folder}")
                    continue

                for fp in sorted(folder.rglob("*")):
                    if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS:
                        continue

                    label, place = get_meta_for_test_file(fp)

                    if label is not None and place is not None:
                        self.files.append(fp)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]

        label, place = get_meta_for_test_file(fp)

        image = Image.open(fp).convert("RGB")
        if self.transform:
            image = self.transform(image)

        group_id = test_group_id_from_label_context(label, place)

        return image, label, place, group_id, fp.stem


# ==========================================
# 5. CORE TRAINING LOGIC
# ==========================================

def ssl_collate_fn(batch):
    num_views = len(batch[0])
    return [torch.stack([s[i] for s in batch]) for i in range(num_views)]


def nt_xent_loss(z1, z2, temperature=0.2):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)

    B = z1.size(0)

    rep = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(rep, rep.T) / temperature

    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    pos = torch.cat([torch.arange(B) + B, torch.arange(B)], dim=0).to(z1.device)

    return F.cross_entropy(sim, pos)


class GroupDROCriterion(nn.Module):
    def __init__(self, num_groups=8, step_size=0.01, warmup_epochs=2, device="cpu"):
        super().__init__()
        self.num_groups = num_groups
        self.step_size = step_size
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.register_buffer("q", torch.ones(num_groups, device=device) / num_groups)
        self.per_sample_ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, group_ids):
        per_sample = self.per_sample_ce(logits, targets)

        group_losses = torch.zeros(self.num_groups, device=logits.device)
        group_counts = torch.zeros(self.num_groups, device=logits.device)

        for g in range(self.num_groups):
            mask = group_ids == g

            if mask.any():
                group_losses[g] = per_sample[mask].mean()
                group_counts[g] = mask.sum()

        observed = group_counts > 0

        if observed.sum() == 0:
            robust_loss = per_sample.mean()
        elif self.current_epoch < self.warmup_epochs:
            robust_loss = group_losses[observed].mean()
        else:
            with torch.no_grad():
                self.q[observed] *= torch.exp(self.step_size * group_losses[observed])
                self.q /= self.q.sum()

            robust_loss = (self.q[observed] * group_losses[observed]).sum()

        return robust_loss, group_losses.detach(), group_counts.detach(), self.q.detach().clone()


# ==========================================
# 6. EVALUATION
# ==========================================

def evaluate_model_train_groups(model, dataloader, device):
    """
    Evaluation for generated validation if needed.
    Uses train contexts:
    grass, outdoor, rock, water.
    """
    model.eval()

    correct, total = 0, 0

    groups = {}
    for label_id in range(NUM_CLASSES):
        for place_id in range(4):
            gid = group_id_from_label_place(label_id, place_id)
            groups[gid] = {
                "name": f"{ID_TO_CLASS[label_id]} {ID_TO_TRAIN_CONTEXT[place_id]}",
                "correct": 0,
                "total": 0,
            }

    with torch.no_grad():
        for images, labels, places, group_ids, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            group_ids = group_ids.to(device)

            preds = torch.argmax(model(images), dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for i in range(labels.size(0)):
                gid = int(group_ids[i].item())

                if gid not in groups:
                    continue

                groups[gid]["total"] += 1

                if preds[i].item() == labels[i].item():
                    groups[gid]["correct"] += 1

    overall_acc = 100 * correct / total if total > 0 else 0.0

    group_accs = {
        k: (
            100 * s["correct"] / s["total"]
            if s["total"] > 0
            else float("nan")
        )
        for k, s in groups.items()
    }

    valid = [a for a in group_accs.values() if not math.isnan(a)]
    worst = min(valid) if valid else 0.0

    return overall_acc, worst, group_accs, groups, correct, total


def evaluate_model_test_groups(model, dataloader, device):
    """
    Evaluation for real test/val.
    Uses test contexts:
    autumn, dim.
    """
    model.eval()

    correct, total = 0, 0

    groups = {}
    for label_id in range(NUM_CLASSES):
        for place_id in range(2):
            gid = test_group_id_from_label_context(label_id, place_id)
            groups[gid] = {
                "name": f"{ID_TO_CLASS[label_id]} {ID_TO_TEST_CONTEXT[place_id]}",
                "correct": 0,
                "total": 0,
            }

    with torch.no_grad():
        for images, labels, places, group_ids, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            group_ids = group_ids.to(device)

            preds = torch.argmax(model(images), dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for i in range(labels.size(0)):
                gid = int(group_ids[i].item())

                if gid not in groups:
                    continue

                groups[gid]["total"] += 1

                if preds[i].item() == labels[i].item():
                    groups[gid]["correct"] += 1

    overall_acc = 100 * correct / total if total > 0 else 0.0

    group_accs = {
        k: (
            100 * s["correct"] / s["total"]
            if s["total"] > 0
            else float("nan")
        )
        for k, s in groups.items()
    }

    valid = [a for a in group_accs.values() if not math.isnan(a)]
    worst = min(valid) if valid else 0.0

    return overall_acc, worst, group_accs, groups, correct, total


# ==========================================
# 7. TRAINING LOOPS
# ==========================================

def train_ssl_model(model, train_loader, device, epochs, lr, weight_decay, temperature, paths):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")

    print(f"\n--- Starting SSL Pretraining ---")

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for views in train_loader:
            v1, v2 = views[0].to(device), views[1].to(device)

            optimizer.zero_grad(set_to_none=True)

            _, z1 = model(v1)
            _, z2 = model(v2)

            loss = nt_xent_loss(z1, z2, temperature=temperature)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)

        print(f"  Epoch [{epoch + 1}/{epochs}] SSL Loss: {avg:.4f}")

        append_row_to_csv(
            paths["stage1_train_log_csv"],
            ["epoch", "ssl_loss"],
            [epoch + 1, f"{avg:.6f}"],
        )

        if avg < best_loss:
            best_loss = avg

            torch.save(model.state_dict(), paths["stage1_best_ssl_model_path"])
            torch.save(model.encoder.state_dict(), paths["stage1_best_encoder_path"])

    torch.save(model.state_dict(), paths["stage1_final_ssl_model_path"])
    torch.save(model.encoder.state_dict(), paths["stage1_final_encoder_path"])

    return best_loss


def train_classifier_model(model, train_loader, val_loader, device, epochs, hp, paths):
    if hp["linear_probe_only"]:
        for p in model.encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
        )

        print("  Stage 2: Linear probe only (encoder frozen)")

    else:
        optimizer = torch.optim.AdamW(
            model.get_param_groups(hp["lr"]),
            weight_decay=hp["weight_decay"],
        )

        print("  Stage 2: Layer-wise fine-tuning")

    groupdro = GroupDROCriterion(
        num_groups=NUM_GROUPS,
        step_size=GROUPDRO_STEP_SIZE,
        warmup_epochs=GROUPDRO_WARMUP_EPOCHS,
        device=device,
    )

    best_wga = -1.0

    print(f"\n--- Starting Stage 2 Training ---")

    for epoch in range(epochs):
        model.train()
        groupdro.current_epoch = epoch

        total_loss, correct, total = 0.0, 0, 0

        for images, labels, places, group_ids, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            group_ids = group_ids.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            robust_loss, batch_gl, batch_gc, q = groupdro(outputs, labels, group_ids)

            robust_loss.backward()
            optimizer.step()

            total_loss += robust_loss.item()

            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(len(train_loader), 1)
        train_acc = 100 * correct / total if total > 0 else 0.0

        print(f"  Epoch [{epoch + 1}/{epochs}] Loss:{avg_loss:.4f}  Acc:{train_acc:.2f}%")

        append_row_to_csv(
            paths["stage2_train_log_csv"],
            ["epoch", "train_loss", "train_acc", "q_values"],
            [
                epoch + 1,
                f"{avg_loss:.6f}",
                f"{train_acc:.2f}",
                json.dumps([float(x) for x in q.detach().cpu().tolist()]),
            ],
        )

        if val_loader:
            overall, worst, group_accs, g_stats, c, t = evaluate_model_test_groups(
                model,
                val_loader,
                device,
            )

            print(f"  -> Val Overall: {overall:.2f}%  Worst-Group: {worst:.2f}%")

            append_row_to_csv(
                paths["stage2_val_log_csv"],
                ["epoch", "val_overall", "val_worst", "val_correct", "val_total", "group_accs"],
                [
                    epoch + 1,
                    f"{overall:.2f}",
                    f"{worst:.2f}",
                    c,
                    t,
                    json.dumps({
                        str(k): None if math.isnan(v) else round(float(v), 4)
                        for k, v in group_accs.items()
                    }),
                ],
            )

            if worst > best_wga:
                best_wga = worst
                torch.save(model.state_dict(), paths["stage2_best_model_path"])

    if not paths["stage2_best_model_path"].exists():
        torch.save(model.state_dict(), paths["stage2_best_model_path"])

    torch.save(model.state_dict(), paths["stage2_final_model_path"])

    return best_wga


# ==========================================
# 8. RUNNER LOGIC
# ==========================================

def get_transforms():
    ssl = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.4),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE * 1.15), int(IMAGE_SIZE * 1.15))),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ssl, train, eval_tf


def print_dataset_summary():
    print("\nDataset roots:")
    print(f"  Generated train root: {TRAIN_ROOT}")
    print(f"  Real nested test root: {TEST_ROOT}")

    print("\nExpected test structure:")
    print(f"  {TEST_ROOT}/autumn/fox")
    print(f"  {TEST_ROOT}/autumn/wolf")
    print(f"  {TEST_ROOT}/dim/fox")
    print(f"  {TEST_ROOT}/dim/wolf")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print_dataset_summary()

    ssl_tf, cls_train_tf, eval_tf = get_transforms()

    all_results = []

    # --- LOOP OVER SEEDS ---
    for current_seed in SEEDS:
        print(f"\n{'=' * 80}\nSTARTING PIPELINE FOR SEED: {current_seed}\n{'=' * 80}")

        set_seed(current_seed)

        seed_dir = EXPERIMENT_DIR / f"seed_{current_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "stage1_train_log_csv": seed_dir / "ssl_train_log.csv",
            "stage1_best_ssl_model_path": seed_dir / "best_ssl_resnet50.pt",
            "stage1_final_ssl_model_path": seed_dir / "final_ssl_resnet50.pt",
            "stage1_best_encoder_path": seed_dir / "best_ssl_encoder.pt",
            "stage1_final_encoder_path": seed_dir / "final_ssl_encoder.pt",

            "stage2_train_log_csv": seed_dir / "classifier_train_log.csv",
            "stage2_val_log_csv": seed_dir / "classifier_val_log.csv",
            "stage2_best_model_path": seed_dir / "best_classifier.pt",
            "stage2_final_model_path": seed_dir / "final_classifier.pt",

            "test_result_csv": seed_dir / "test_result.csv",
            "config_json": seed_dir / "config.json",
        }

        save_json(paths["config_json"], {
            "experiment_name": EXPERIMENT_NAME,
            "seed": current_seed,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "num_groups": NUM_GROUPS,
            "classes": CLASS_TO_ID,
            "train_contexts": TRAIN_CONTEXT_TO_ID,
            "test_contexts": TEST_CONTEXT_TO_ID,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "ssl_projection_dim": SSL_PROJECTION_DIM,
            "ssl_hidden_dim": SSL_HIDDEN_DIM,
            "ssl_temperature": SSL_TEMPERATURE,
            "ssl_lr": SSL_LR,
            "ssl_weight_decay": SSL_WEIGHT_DECAY,
            "cls_lr": CLS_LR,
            "cls_weight_decay": CLS_WEIGHT_DECAY,
            "cls_dropout": CLS_DROPOUT,
            "train_root": str(TRAIN_ROOT),
            "test_root": str(TEST_ROOT),
            "val_from_test_split": "15%",
            "final_test_from_test_split": "85%",
        })

        # ==========================================
        # Dataloaders
        # ==========================================

        ssl_ds = NicoMultiViewSSLDataset(
            TRAIN_ROOT,
            transform=ssl_tf,
            num_views=SSL_NUM_VIEWS,
        )

        if len(ssl_ds) == 0:
            raise RuntimeError(
                "SSL dataset has 0 valid groups. "
                "Check generated CSV sidecars and output_image paths."
            )

        weights, ssl_counts = ssl_ds.get_sample_weights()

        print("\nSSL group counts:")
        for gid, count in sorted(ssl_counts.items()):
            cls_id = gid // 4
            ctx_id = gid % 4
            print(f"  g{gid}: {ID_TO_CLASS[cls_id]} {ID_TO_TRAIN_CONTEXT[ctx_id]} -> {count}")

        ssl_loader = DataLoader(
            ssl_ds,
            batch_size=BATCH_SIZE,
            drop_last=True,
            collate_fn=ssl_collate_fn,
            sampler=WeightedRandomSampler(
                torch.DoubleTensor(weights),
                len(weights),
                replacement=True,
            ),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        full_train_ds = NicoGeneratedClassificationDataset(
            TRAIN_ROOT,
            transform=cls_train_tf,
        )

        if len(full_train_ds) == 0:
            raise RuntimeError(
                "Classification train dataset has 0 valid samples. "
                "Check generated CSV sidecars."
            )

        train_group_ids = full_train_ds.group_ids
        train_counts = Counter(train_group_ids)

        print("\nTrain group counts:")
        for gid, count in sorted(train_counts.items()):
            cls_id = gid // 4
            ctx_id = gid % 4
            print(f"  g{gid}: {ID_TO_CLASS[cls_id]} {ID_TO_TRAIN_CONTEXT[ctx_id]} -> {count}")

        train_weights = [1.0 / train_counts[g] for g in train_group_ids]

        train_loader = DataLoader(
            full_train_ds,
            batch_size=BATCH_SIZE,
            sampler=WeightedRandomSampler(
                torch.DoubleTensor(train_weights),
                len(train_weights),
                replacement=True,
            ),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        full_test_ds = NicoNestedTestDataset(
            TEST_ROOT,
            transform=eval_tf,
        )

        if len(full_test_ds) == 0:
            raise RuntimeError(
                "Test dataset has 0 valid samples. "
                "Expected test/autumn/fox, test/autumn/wolf, test/dim/fox, test/dim/wolf."
            )

        # Same as your original code:
        # 85% final test, 15% validation from real test split.
        test_size = int(0.85 * len(full_test_ds))
        val_size  = len(full_test_ds) - test_size

        test_subset, val_subset = random_split(
            full_test_ds,
            [test_size, val_size],
            generator=torch.Generator().manual_seed(current_seed),
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ) if len(test_subset) > 0 else None

        print(f"\nReal test split:")
        print(f"  Full real test samples: {len(full_test_ds)}")
        print(f"  Validation samples: {len(val_subset)}")
        print(f"  Final test samples: {len(test_subset)}")

        # ==========================================
        # Stage 1: SSL Pretraining
        # ==========================================

        stage1_model = SSLResNet50(
            projection_dim=SSL_PROJECTION_DIM,
            projector_hidden_dim=SSL_HIDDEN_DIM,
            pretrained_backbone=SSL_START_FROM_IMAGENET,
        ).to(device)

        if not paths["stage1_best_encoder_path"].exists():
            print(f"\n[Stage 1: SSL PRETRAINING] Hidden Dim: {SSL_HIDDEN_DIM}")

            train_ssl_model(
                stage1_model,
                ssl_loader,
                device,
                STAGE1_EPOCHS,
                SSL_LR,
                SSL_WEIGHT_DECAY,
                SSL_TEMPERATURE,
                paths,
            )

        else:
            print(f"\n[Stage 1: CACHE HIT] Found existing SSL encoder for seed {current_seed}. Skipping Stage 1.")

        # ==========================================
        # Stage 2: Classifier Training
        # ==========================================

        print(f"\n[Stage 2: CLASSIFIER TRAINING]")

        stage2_model = ResNet50Classifier(
            num_classes=NUM_CLASSES,
            dropout=CLS_DROPOUT,
            pretrained_backbone=False,
        ).to(device)

        if STAGE2_START_FROM_SSL and paths["stage1_best_encoder_path"].exists():
            stage2_model.encoder.load_state_dict(
                torch.load(paths["stage1_best_encoder_path"], map_location=device)
            )
            print(f"Loaded cached SSL encoder from: {paths['stage1_best_encoder_path']}")

        cls_hp = {
            "lr": CLS_LR,
            "weight_decay": CLS_WEIGHT_DECAY,
            "dropout": CLS_DROPOUT,
            "linear_probe_only": CLS_LINEAR_PROBE_ONLY,
        }

        best_val_wga = train_classifier_model(
            stage2_model,
            train_loader,
            val_loader,
            device,
            STAGE2_EPOCHS,
            cls_hp,
            paths,
        )

        # ==========================================
        # Test Evaluation
        # ==========================================

        test_overall, test_worst = float("nan"), float("nan")

        if test_loader and paths["stage2_best_model_path"].exists():
            stage2_model.load_state_dict(
                torch.load(paths["stage2_best_model_path"], map_location=device)
            )

            test_overall, test_worst, test_group_accs, test_groups, correct, total = evaluate_model_test_groups(
                stage2_model,
                test_loader,
                device,
            )

            print(f"  -> Test Overall: {test_overall:.2f}%  |  Test WGA: {test_worst:.2f}%")

            append_row_to_csv(
                paths["test_result_csv"],
                [
                    "seed",
                    "test_overall",
                    "test_wga",
                    "correct",
                    "total",
                    "group_accs",
                ],
                [
                    current_seed,
                    f"{test_overall:.2f}",
                    f"{test_worst:.2f}",
                    correct,
                    total,
                    json.dumps({
                        str(k): None if math.isnan(v) else round(float(v), 4)
                        for k, v in test_group_accs.items()
                    }),
                ],
            )

        # ==========================================
        # Logging
        # ==========================================

        append_row_to_csv(
            MASTER_SUMMARY_CSV,
            [
                "seed",
                "ssl_hidden_dim",
                "ssl_projection_dim",
                "lr",
                "wd",
                "dropout",
                "linear_probe",
                "best_val_wga",
                "test_overall",
                "test_wga",
                "dir",
            ],
            [
                current_seed,
                SSL_HIDDEN_DIM,
                SSL_PROJECTION_DIM,
                CLS_LR,
                CLS_WEIGHT_DECAY,
                CLS_DROPOUT,
                CLS_LINEAR_PROBE_ONLY,
                f"{best_val_wga:.2f}",
                f"{test_overall:.2f}",
                f"{test_worst:.2f}",
                str(seed_dir),
            ],
        )

        all_results.append({
            "seed": current_seed,
            "val_wga": best_val_wga,
            "test_overall": test_overall,
            "test_wga": test_worst,
        })

    # ==========================================
    # Final Report
    # ==========================================

    print("\n" + "#" * 90 + "\nALL RUNS COMPLETED\n" + "#" * 90)

    for r in all_results:
        print(
            f"Seed: {r['seed']} | "
            f"best_val_wga: {r['val_wga']:.2f} | "
            f"test_overall: {r['test_overall']:.2f} | "
            f"test_wga: {r['test_wga']:.2f}"
        )


if __name__ == "__main__":
    main()
