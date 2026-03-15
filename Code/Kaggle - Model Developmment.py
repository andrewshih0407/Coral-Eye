import os, random, warnings, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = True

DATA_DIR_1   = "/kaggle/input/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification"
DATA_DIR_2   = "/kaggle/input/datasets/sonainjamil/bleached-corals-detection/Train"
DATA_DIR_3   = "/kaggle/input/datasets/sonainjamil/bhd-corals/Dataset"
MERGED_DIR   = "/kaggle/working/merged_dataset"  
IMG_SIZE     = 260
BATCH_SIZE   = 32
EPOCHS       = 60
LR_HEAD      = 5e-4
LR_FINE      = 8e-5
FINE_EPOCH   = 10        
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
TTA_STEPS    = 8
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES      = ["bleached", "healthy"]  

print(f"Device : {DEVICE}")
=
def safe_copy(src_glob_pattern, dst_dir, prefix=""):
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(src_glob_pattern, recursive=True)
    copied = 0
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        dst = os.path.join(dst_dir, f"{prefix}{os.path.basename(f)}")
        # avoid name collision
        if os.path.exists(dst):
            base, ext2 = os.path.splitext(dst)
            dst = f"{base}_{random.randint(1000,9999)}{ext2}"
        shutil.copy2(f, dst)
        copied += 1
    return copied

bleached_dst = os.path.join(MERGED_DIR, "bleached")
healthy_dst  = os.path.join(MERGED_DIR, "healthy")

n = safe_copy(os.path.join(DATA_DIR_1, "bleached_corals", "**", "*"), bleached_dst, "ds1_")
print(f"DS1 bleached copied : {n}")
n = safe_copy(os.path.join(DATA_DIR_1, "healthy_corals",  "**", "*"), healthy_dst,  "ds1_")
print(f"DS1 healthy  copied : {n}")

n = safe_copy(os.path.join(DATA_DIR_2, "Bleached",   "**", "*"), bleached_dst, "ds2_")
print(f"DS2 bleached copied : {n}")
n = safe_copy(os.path.join(DATA_DIR_2, "Unbleached", "**", "*"), healthy_dst,  "ds2_")
print(f"DS2 healthy  copied : {n}")

n = safe_copy(os.path.join(DATA_DIR_3, "Bleached", "**", "*"), bleached_dst, "ds3_")
print(f"DS3 bleached copied : {n}")
n = safe_copy(os.path.join(DATA_DIR_3, "Healthy",  "**", "*"), healthy_dst,  "ds3_")
print(f"DS3 healthy  copied : {n}")
print("DS3 Dead coral       : skipped (not relevant to binary task)")

total_b = len(os.listdir(bleached_dst))
total_h = len(os.listdir(healthy_dst))
print(f"\nMerged  bleached: {total_b}  |  healthy: {total_h}  |  total: {total_b+total_h}")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tta_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset; self.indices = indices; self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label

class RawSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.dataset[self.indices[i]]

full_dataset = datasets.ImageFolder(root=MERGED_DIR, transform=None)
print(f"\nClasses : {full_dataset.classes}")
print(f"Total   : {len(full_dataset)} images")

labels  = [s[1] for s in full_dataset.samples]
all_idx = list(range(len(full_dataset)))

train_idx, temp_idx, y_train, y_temp = train_test_split(
    all_idx, labels,
    test_size=(VAL_SPLIT + TEST_SPLIT), stratify=labels, random_state=SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
    stratify=y_temp, random_state=SEED
)
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

train_ds = TransformSubset(full_dataset, train_idx, train_tf)
val_ds   = TransformSubset(full_dataset, val_idx,   val_tf)
test_ds  = TransformSubset(full_dataset, test_idx,  val_tf)
test_raw = RawSubset(full_dataset, test_idx)

train_labels  = [labels[i] for i in train_idx]
class_counts  = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_w      = [class_weights[l] for l in train_labels]
sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,    num_workers=2, pin_memory=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, label_smoothing=self.ls, reduction="none")
        pt   = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    rw = int(W * np.sqrt(1-lam)); rh = int(H * np.sqrt(1-lam))
    cx = np.random.randint(W);    cy = np.random.randint(H)
    x1,x2 = np.clip(cx-rw//2,0,W), np.clip(cx+rw//2,0,W)
    y1,y2 = np.clip(cy-rh//2,0,H), np.clip(cy+rh//2,0,H)
    mx = x.clone(); mx[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(W*H)
    return mx, y, y[idx], lam

def mixed_loss(crit, pred, ya, yb, lam):
    return lam*crit(pred,ya) + (1-lam)*crit(pred,yb)

def build_model(num_classes=2):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.parameters(): p.requires_grad = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Dropout(p=0.5),
        nn.Linear(in_f, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return m

model     = build_model().to(DEVICE)
criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)

def run_epoch(loader, train=True, use_aug=False):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            if train and use_aug:
                fn = mixup if random.random() < 0.5 else cutmix
                imgs, ya, yb, lam = fn(imgs, lbls)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = mixed_loss(criterion, out, ya, yb, lam)
            else:
                if train: optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, lbls)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == lbls).sum().item()
            total      += imgs.size(0)
    return total_loss/total, correct/total

history   = {"tl":[], "vl":[], "ta":[], "va":[]}
best_val  = 0.0
ckpt      = "/kaggle/working/best_v4.pth"

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_HEAD, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR_HEAD,
    steps_per_epoch=len(train_loader), epochs=FINE_EPOCH, pct_start=0.3)

print("\n── Phase 1: Head only ────────────────────────────────")
for ep in range(1, FINE_EPOCH+1):
    tl,ta = run_epoch(train_loader, train=True,  use_aug=False)
    vl,va = run_epoch(val_loader,   train=False)
    history["tl"].append(tl); history["vl"].append(vl)
    history["ta"].append(ta); history["va"].append(va)
    flag = ""
    if va > best_val: best_val=va; torch.save(model.state_dict(), ckpt); flag=" ✓"
    print(f"  Ep {ep:02d}/{EPOCHS}  tr={ta:.4f}  val={va:.4f}{flag}")

print("\n── Phase 2: Full fine-tune + MixUp/CutMix ───────────")
for p in model.parameters(): p.requires_grad = True

remaining = EPOCHS - FINE_EPOCH
optimizer = optim.AdamW(model.parameters(), lr=LR_FINE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6)

for ep in range(FINE_EPOCH+1, EPOCHS+1):
    tl,ta = run_epoch(train_loader, train=True,  use_aug=True)
    vl,va = run_epoch(val_loader,   train=False)
    history["tl"].append(tl); history["vl"].append(vl)
    history["ta"].append(ta); history["va"].append(va)
    flag = ""
    if va > best_val: best_val=va; torch.save(model.state_dict(), ckpt); flag=" ✓"
    print(f"  Ep {ep:02d}/{EPOCHS}  tr={ta:.4f}  val={va:.4f}{flag}")

print(f"\nBest val accuracy: {best_val:.4f}")

print("\n── TTA Inference ─────────────────────────────────────")
model.load_state_dict(torch.load(ckpt))
model.eval()

tta_probs, tta_labels = [], []
with torch.no_grad():
    for pil_img, label in test_raw:
        probs = []
        for _ in range(TTA_STEPS):
            t = tta_tf(pil_img).unsqueeze(0).to(DEVICE)
            probs.append(torch.softmax(model(t), dim=1)[0].cpu().numpy())
        t = val_tf(pil_img).unsqueeze(0).to(DEVICE)
        probs.append(torch.softmax(model(t), dim=1)[0].cpu().numpy())
        tta_probs.append(np.mean(probs, axis=0))
        tta_labels.append(label)

tta_probs  = np.array(tta_probs)  
tta_labels = np.array(tta_labels)
tta_preds  = tta_probs.argmax(axis=1)

print("\n── Test Results (TTA) ────────────────────────────────")
print(classification_report(tta_labels, tta_preds,
                             target_names=full_dataset.classes, digits=4))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("PaleWatch v4 — EfficientNetB0 Results (3 Datasets)", fontsize=14, fontweight="bold")

axes[0,0].plot(history["tl"], label="Train", color="#E05C5C")
axes[0,0].plot(history["vl"], label="Val",   color="#5C9EE0")
axes[0,0].axvline(FINE_EPOCH-0.5, color="gray", linestyle=":", label="Unfreeze")
axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].set_xlabel("Epoch")

axes[0,1].plot(history["ta"], label="Train", color="#E05C5C")
axes[0,1].plot(history["va"], label="Val",   color="#5C9EE0")
axes[0,1].axhline(0.95, color="green", linestyle="--", lw=1.2, label="95% target")
axes[0,1].axvline(FINE_EPOCH-0.5, color="gray", linestyle=":", label="Unfreeze")
axes[0,1].set_title("Accuracy"); axes[0,1].legend(); axes[0,1].set_xlabel("Epoch")

cm = confusion_matrix(tta_labels, tta_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes, ax=axes[1,0])
axes[1,0].set_title("Confusion Matrix (TTA)")
axes[1,0].set_ylabel("True"); axes[1,0].set_xlabel("Predicted")

pos_scores = tta_probs[:, 1]  

fpr, tpr, _ = roc_curve(tta_labels, pos_scores, pos_label=1)
roc_auc     = auc(fpr, tpr)

fpr0, tpr0, _ = roc_curve((tta_labels == 0).astype(int), tta_probs[:, 0])
auc0           = auc(fpr0, tpr0)
fpr1, tpr1, _ = roc_curve((tta_labels == 1).astype(int), tta_probs[:, 1])
auc1           = auc(fpr1, tpr1)

axes[1,1].plot(fpr0, tpr0, color="#E05C5C", lw=2,
               label=f"{full_dataset.classes[0]} (AUC={auc0:.3f})")
axes[1,1].plot(fpr1, tpr1, color="#5C9EE0", lw=2,
               label=f"{full_dataset.classes[1]} (AUC={auc1:.3f})")
axes[1,1].plot([0,1],[0,1], "gray", lw=1, linestyle=":")
axes[1,1].set_title(f"ROC Curve  |  Macro AUC = {(auc0+auc1)/2:.3f}")
axes[1,1].set_xlabel("False Positive Rate")
axes[1,1].set_ylabel("True Positive Rate")
axes[1,1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("/kaggle/working/palewatch_v4_results.png", dpi=150)
plt.show()
print("Saved → palewatch_v4_results.png")

torch.save({
    "model_state_dict" : model.state_dict(),
    "class_to_idx"     : full_dataset.class_to_idx,
    "img_size"         : IMG_SIZE,
    "architecture"     : "efficientnet_b0_v4",
    "best_val_acc"     : best_val,
    "macro_auc"        : (auc0+auc1)/2,
}, "/kaggle/working/palewatch_v4_final.pth")
print(f"Saved  |  best_val={best_val:.4f}  |  macro_auc={(auc0+auc1)/2:.4f}")

def predict_image(img_path, tta=True):
    img = Image.open(img_path).convert("RGB")
    model.eval()
    with torch.no_grad():
        if tta:
            probs = [torch.softmax(model(tta_tf(img).unsqueeze(0).to(DEVICE)),1)[0].cpu().numpy()
                     for _ in range(TTA_STEPS)]
            probs.append(torch.softmax(model(val_tf(img).unsqueeze(0).to(DEVICE)),1)[0].cpu().numpy())
            prob = np.mean(probs, axis=0)
        else:
            prob = torch.softmax(model(val_tf(img).unsqueeze(0).to(DEVICE)),1)[0].cpu().numpy()
    idx = prob.argmax()
    print(f"→ {full_dataset.classes[idx]}  ({prob[idx]:.2%} confidence)")
    return full_dataset.classes[idx], prob
