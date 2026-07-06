import os
import re
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
# 0. EXPERIMENT CONFIG & PARAMS
# ==========================================

EXPERIMENT_NAME = "metashift"

# ---------- General ----------
BATCH_SIZE   = 64
NUM_WORKERS  = 4
IMAGE_SIZE   = 224
NUM_CLASSES  = 2
NUM_GROUPS   = 4

# RUN OVER 5 SEEDS
SEEDS        = [42, 43, 44]  # You can modify this list to run more or fewer seeds

# ---------- Stage 1 SSL ----------
STAGE1_EPOCHS            = 10
SSL_TEMPERATURE          = 0.2
SSL_PROJECTION_DIM       = 128
SSL_START_FROM_IMAGENET  = True
SSL_NUM_VIEWS            = 2

SSL_LR           = 1e-4
SSL_WEIGHT_DECAY = 1e-4
SSL_HIDDEN_DIM   = 512  # Fixed

# ---------- Stage 2 Classifier ----------
STAGE2_EPOCHS            = 6
STAGE2_START_FROM_SSL    = True
GROUPDRO_WARMUP_EPOCHS   = 2
GROUPDRO_STEP_SIZE       = 0.01

# FIXED CLASSIFIER HYPERPARAMETERS
CLASSIFIER_LR            = 1e-4
CLASSIFIER_WD            = 1e-4
CLASSIFIER_DROPOUT       = 0.5
CLASSIFIER_LINEAR_PROBE  = False

# ---------- Data Roots ----------
TRAIN_ROOT   = REPO_ROOT / "data" / "metashifts" / "generated" / "train"
TEST_CAT_DIR = REPO_ROOT / "data" / "metashifts" / "metashift_original" / "test" / "cat" / "cat(shelf)"
TEST_DOG_DIR = REPO_ROOT / "data" / "metashifts" / "metashift_original" / "test" / "dog" / "dog(shelf)"

# Setup Master Directory
EXPERIMENT_DIR = REPO_ROOT / "output" / EXPERIMENT_NAME
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_SUMMARY_CSV = EXPERIMENT_DIR / "master_summary.csv"

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
    def __init__(self, projection_dim=128, projector_hidden_dim=2048, pretrained_backbone=False):
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
        """Layer-wise LR for fine-tuning"""
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ==========================================
# 3. METADATA HELPERS (MetaShift Specific)
# ==========================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def load_sidecar_csv(csv_path: Path):
    data = {}
    if not csv_path.exists():
        return data
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        for row in reader:
            if len(row) >= 2:
                data[row[0].strip()] = row[1].strip()
    return data

def label_from_class_name(class_name: str):
    c = str(class_name).strip().lower()
    if c == "cat": return 0
    if c == "dog": return 1
    return None

def place_from_context(context: str):
    """Maps dataset contexts to 0 (Indoor) or 1 (Outdoor)."""
    c = str(context).strip().lower()
    if "bed" in c or "sofa" in c or "shelf" in c:
        return 0
    if "bench" in c or "bike" in c:
        return 1
    return None

def group_id_from_label_place(label, place):
    return int(label) * 2 + int(place)

def get_meta_for_generated_training_file(file_path: Path):
    sidecar_path = file_path.with_suffix(".csv")
    if not sidecar_path.exists(): return None, None
    sidecar = load_sidecar_csv(sidecar_path)
    label = label_from_class_name(sidecar.get("class_name", ""))
    
    place = None
    if sidecar.get("type") == "generated_image":
        var_idx = sidecar.get("variation_index")
        if var_idx in ["0", "1"]:
            place = 0 
        elif var_idx in ["2", "3"]:
            place = 1
    elif sidecar.get("type") == "original_image":
        place = place_from_context(sidecar.get("source_context", ""))

    if label is None or place is None: return None, None
    return label, place

def get_meta_for_test_file(file_path: Path):
    path_text = str(file_path).lower()
    label = 0 if "/cat/" in path_text else (1 if "/dog/" in path_text else None)
    place = place_from_context(path_text)
    if label is None or place is None: return None, None
    return label, place

def get_group_key(stem: str) -> str:
    m = re.match(r"^(\d+)_(orig|var\d+)$", stem)
    return m.group(1) if m else stem

# ==========================================
# 4. DATASETS
# ==========================================

class MetaShiftMultiViewSSLDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_views=2):
        self.transform = transform
        self.num_views = num_views
        root = Path(root_dir)

        groups, group_meta = defaultdict(list), {}
        for fp in sorted(root.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS: continue
            if not re.match(r"^\d+_(orig|var\d+)$", fp.stem): continue
            label, place = get_meta_for_generated_training_file(fp)
            if label is None: continue
            key = get_group_key(fp.stem)
            groups[key].append(fp)
            group_meta[key] = (label, place)

        self.groups, self.group_ids = [], []
        for key, files in groups.items():
            if len(files) < 2: continue
            label, place = group_meta[key]
            self.groups.append(files)
            self.group_ids.append(group_id_from_label_place(label, place))

    def __len__(self): return len(self.groups)

    def __getitem__(self, idx):
        chosen = random.sample(self.groups[idx], k=min(self.num_views, len(self.groups[idx])))
        views = []
        for path in chosen:
            img = Image.open(path).convert("RGB")
            views.append(self.transform(img) if self.transform else transforms.ToTensor()(img))
        return views

    def get_sample_weights(self):
        counts = Counter(self.group_ids)
        return [1.0 / counts[g] for g in self.group_ids], counts

class MetaShiftGeneratedClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        root = Path(root_dir)

        self.files, self.group_ids = [], []
        for fp in sorted(root.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS: continue
            label, place = get_meta_for_generated_training_file(fp)
            if label is None: continue
            self.files.append(fp)
            self.group_ids.append(group_id_from_label_place(label, place))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        label, place = get_meta_for_generated_training_file(fp)
        image = Image.open(fp).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label, place, group_id_from_label_place(label, place), fp.stem

class DatasetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        img, label, place, group_id, stem = self.subset[idx]
        if self.transform: img = self.transform(img)
        return img, label, place, group_id, stem
    def __len__(self): return len(self.subset)

class MetaShiftShelfTestDataset(Dataset):
    def __init__(self, cat_dir, dog_dir, transform=None):
        self.transform = transform
        self.files = []
        for folder in [Path(cat_dir), Path(dog_dir)]:
            for fp in sorted(folder.rglob("*")):
                if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS: continue
                label, place = get_meta_for_test_file(fp)
                if label is not None: self.files.append(fp)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        label, place = get_meta_for_test_file(fp)
        image = Image.open(fp).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label, place, group_id_from_label_place(label, place), fp.stem

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
    def __init__(self, num_groups=4, step_size=0.01, warmup_epochs=2, device="cpu"):
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
# 6. EVALUATION
# ==========================================

def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    groups = {
        (0, 0): {"name": "Cat Indoor",   "correct": 0, "total": 0},
        (0, 1): {"name": "Cat Outdoor",  "correct": 0, "total": 0},
        (1, 0): {"name": "Dog Indoor",   "correct": 0, "total": 0},
        (1, 1): {"name": "Dog Outdoor",  "correct": 0, "total": 0},
    }
    with torch.no_grad():
        for images, labels, places, _, _ in dataloader:
            images, labels, places = images.to(device), labels.to(device), places.to(device)
            preds = torch.argmax(model(images), dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for i in range(labels.size(0)):
                lbl, plc = labels[i].item(), places[i].item()
                groups[(lbl, plc)]["total"] += 1
                if preds[i].item() == lbl:
                    groups[(lbl, plc)]["correct"] += 1

    overall_acc = 100 * correct / total if total > 0 else 0.0
    group_accs = {k: (100 * s["correct"] / s["total"] if s["total"] > 0 else float("nan")) for k, s in groups.items()}
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
            optimizer.zero_grad()
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  Epoch [{epoch+1}/{epochs}] SSL Loss: {avg:.4f}")
        append_row_to_csv(paths["stage1_train_log_csv"], ["epoch", "ssl_loss"], [epoch + 1, f"{avg:.6f}"])

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), paths["stage1_best_ssl_model_path"])
            torch.save(model.encoder.state_dict(), paths["stage1_best_encoder_path"])

    torch.save(model.state_dict(), paths["stage1_final_ssl_model_path"])
    torch.save(model.encoder.state_dict(), paths["stage1_final_encoder_path"])
    return best_loss

def train_classifier_model(model, train_loader, val_loader, device, epochs, linear_probe, lr, weight_decay, paths):
    if linear_probe:
        for p in model.encoder.parameters(): p.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        print("  Stage 2: Linear probe only (encoder frozen)")
    else:
        optimizer = torch.optim.AdamW(model.get_param_groups(lr), weight_decay=weight_decay)
        print("  Stage 2: Layer-wise fine-tuning")

    groupdro = GroupDROCriterion(num_groups=NUM_GROUPS, step_size=GROUPDRO_STEP_SIZE, warmup_epochs=GROUPDRO_WARMUP_EPOCHS, device=device)
    best_wga = -1.0

    print(f"\n--- Starting Stage 2 Training ---")
    for epoch in range(epochs):
        model.train()
        groupdro.current_epoch = epoch
        total_loss, correct, total = 0.0, 0, 0

        for images, labels, places, group_ids, _ in train_loader:
            images, labels, group_ids = images.to(device), labels.to(device), group_ids.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            robust_loss, batch_gl, batch_gc, q = groupdro(outputs, labels, group_ids)
            robust_loss.backward()
            optimizer.step()
            
            total_loss += robust_loss.item()
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(len(train_loader), 1)
        train_acc = 100 * correct / total if total > 0 else 0.0
        print(f"  Epoch [{epoch+1}/{epochs}] Loss:{avg_loss:.4f}  Acc:{train_acc:.2f}%")

        if val_loader:
            overall, worst, group_accs, g_stats, c, t = evaluate_model(model, val_loader, device)
            print(f"  -> Val Overall: {overall:.2f}%  Worst-Group: {worst:.2f}%")
            if worst > best_wga:
                best_wga = worst
                torch.save(model.state_dict(), paths["stage2_best_model_path"])

    if not paths["stage2_best_model_path"].exists():
        torch.save(model.state_dict(), paths["stage2_best_model_path"])
        
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ssl_tf, cls_train_tf, eval_tf = get_transforms()
    all_results = []
    
    # Global datasets setup
    full_test_ds = MetaShiftShelfTestDataset(TEST_CAT_DIR, TEST_DOG_DIR, transform=None)
    train_ds_raw = MetaShiftGeneratedClassificationDataset(TRAIN_ROOT, transform=cls_train_tf)
    train_group_ids = train_ds_raw.group_ids
    train_counts = Counter(train_group_ids)
    train_weights = [1.0 / train_counts[g] for g in train_group_ids]
    
    for seed in SEEDS:
        print(f"\n{'='*60}\n[RUNNING SEED: {seed}]\n{'='*60}")
        set_seed(seed) 
        
        # Directory setup for the current seed
        seed_dir = EXPERIMENT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Prepare SSL Dataloader
        ssl_ds = MetaShiftMultiViewSSLDataset(TRAIN_ROOT, transform=ssl_tf, num_views=SSL_NUM_VIEWS)
        weights, _ = ssl_ds.get_sample_weights()
        ssl_loader = DataLoader(ssl_ds, batch_size=BATCH_SIZE, drop_last=True, collate_fn=ssl_collate_fn,
                                sampler=WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), True))

        # 2. Prepare Train Dataloader
        train_loader = DataLoader(
            train_ds_raw, batch_size=BATCH_SIZE, 
            sampler=WeightedRandomSampler(torch.DoubleTensor(train_weights), len(train_weights), True),
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
        )

        # 3. Prepare Val and Test Dataloaders (split 15/85 from original Test set)
        val_loader, test_loader = None, None
        if len(full_test_ds) > 0:
            val_size = int(0.15 * len(full_test_ds))
            test_size = len(full_test_ds) - val_size
            
            # Split varies slightly per seed if initialized with the seeded generator
            test_subset, val_subset = random_split(
                full_test_ds, [test_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            
            val_ds = DatasetWithTransform(val_subset, transform=eval_tf)
            test_ds = DatasetWithTransform(test_subset, transform=eval_tf)
            
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
        print(f"  Stage 2 Train Samples: {len(train_ds_raw)}")
        print(f"  Stage 2 Val Samples:   {len(val_subset) if val_loader else 0}")
        print(f"  Stage 2 Test Samples:  {len(test_subset) if test_loader else 0}")

        # --- STAGE 1: SSL PRETRAINING ---
        stage1_paths = {
            "stage1_train_log_csv": seed_dir / "ssl_train_log.csv",
            "stage1_best_ssl_model_path": seed_dir / "best_ssl_resnet50.pt",
            "stage1_final_ssl_model_path": seed_dir / "final_ssl_resnet50.pt",
            "stage1_best_encoder_path": seed_dir / "best_ssl_encoder.pt",
            "stage1_final_encoder_path": seed_dir / "final_ssl_encoder.pt",
        }
        
        stage1_model = SSLResNet50(projection_dim=SSL_PROJECTION_DIM, projector_hidden_dim=SSL_HIDDEN_DIM, pretrained_backbone=SSL_START_FROM_IMAGENET).to(device)
        
        if not stage1_paths["stage1_best_encoder_path"].exists():
            train_ssl_model(stage1_model, ssl_loader, device, STAGE1_EPOCHS, SSL_LR, SSL_WEIGHT_DECAY, SSL_TEMPERATURE, stage1_paths)
        else:
            print(f"\n[CACHE HIT] Found existing SSL encoder for seed {seed}. Skipping Stage 1.")

        # --- STAGE 2: CLASSIFIER ---
        stage2_paths = {
            "stage2_best_model_path": seed_dir / "best_classifier.pt",
            "stage2_final_model_path": seed_dir / "final_classifier.pt",
        }
        
        stage2_model = ResNet50Classifier(num_classes=NUM_CLASSES, dropout=CLASSIFIER_DROPOUT, pretrained_backbone=False).to(device)
        
        if STAGE2_START_FROM_SSL and stage1_paths["stage1_best_encoder_path"].exists():
            stage2_model.encoder.load_state_dict(torch.load(stage1_paths["stage1_best_encoder_path"], map_location=device))
            print(f"Loaded cached SSL encoder for seed {seed}.")

        best_val_wga = train_classifier_model(
            stage2_model, train_loader, val_loader, device, STAGE2_EPOCHS, 
            CLASSIFIER_LINEAR_PROBE, CLASSIFIER_LR, CLASSIFIER_WD, stage2_paths
        )
        
        # Test Evaluation
        test_overall, test_worst = float("nan"), float("nan")
        if test_loader and stage2_paths["stage2_best_model_path"].exists():
            stage2_model.load_state_dict(torch.load(stage2_paths["stage2_best_model_path"], map_location=device))
            test_overall, test_worst, _, _, _, _ = evaluate_model(stage2_model, test_loader, device)
            print(f"  -> Test Overall: {test_overall:.2f}%  |  Test WGA: {test_worst:.2f}%")

        # Log to Master CSV
        append_row_to_csv(MASTER_SUMMARY_CSV, 
            ["seed", "ssl_dim", "lr", "wd", "dropout", "linear_probe", "best_val_wga", "test_overall", "test_wga", "dir"],
            [seed, SSL_HIDDEN_DIM, CLASSIFIER_LR, CLASSIFIER_WD, CLASSIFIER_DROPOUT, CLASSIFIER_LINEAR_PROBE, f"{best_val_wga:.2f}", f"{test_overall:.2f}", f"{test_worst:.2f}", str(seed_dir)]
        )
        
        all_results.append({"seed": seed, "val_wga": best_val_wga, "test_wga": test_worst})

    print("\n" + "#" * 90 + "\nALL RUNS COMPLETED\n" + "#" * 90)
    for r in all_results: 
        print(f"seed={r['seed']} | best_val_wga={r['val_wga']:.2f} | test_wga={r['test_wga']:.2f}")

if __name__ == "__main__":
    main()