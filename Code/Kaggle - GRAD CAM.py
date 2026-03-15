import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import random

MERGED_DIR  = "/kaggle/working/merged_dataset"
CKPT        = "/kaggle/working/best_v4.pth"
IMG_SIZE    = 260
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SHOW      = 4

def build_model(num_classes=2):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Dropout(0.5),
        nn.Linear(in_f, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return m

model = build_model().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

full_dataset = datasets.ImageFolder(root=MERGED_DIR, transform=None)
CLASS_NAMES  = full_dataset.classes

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx, probs[class_idx].item()

target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

def overlay_cam(pil_img, cam, alpha=0.45):
    img_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)/255.0
    heatmap = cm.jet(cam)[:, :, :3]
    blended = alpha * heatmap + (1 - alpha) * img_np
    return np.clip(blended, 0, 1)

samples_by_class = {0: [], 1: []}
for path, label in full_dataset.samples:
    samples_by_class[label].append(path)

random.seed(99)
selected = {cls_idx: random.sample(paths, min(N_SHOW, len(paths)))
            for cls_idx, paths in samples_by_class.items()}

n_classes = len(CLASS_NAMES)
fig, axes = plt.subplots(n_classes*2, N_SHOW, figsize=(N_SHOW*3.5, n_classes*2*3.2))
fig.suptitle("Grad-CAM — What the model looks at\n(top: original | bottom: heatmap)",
             fontsize=13, fontweight="bold", y=1.01)

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    for col, img_path in enumerate(selected[cls_idx]):
        pil_img = Image.open(img_path).convert("RGB")
        tensor = val_tf(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)
        cam, pred_idx, confidence = grad_cam.generate(tensor)
        cam_resized = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))/255.0
        overlay = overlay_cam(pil_img, cam_resized)
        row_orig = cls_idx*2
        row_cam  = cls_idx*2+1
        axes[row_orig][col].imshow(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        axes[row_orig][col].axis("off")
        pred_label = CLASS_NAMES[pred_idx]
        correct = "✓" if pred_idx==cls_idx else "✗"
        color = "green" if pred_idx==cls_idx else "red"
        axes[row_orig][col].set_title(f"True: {cls_name}\nPred: {pred_label} {correct} ({confidence:.0%})",
                                      fontsize=8, color=color)
        if col==0:
            axes[row_orig][col].set_ylabel(f"{cls_name.upper()}\nOriginal", fontsize=9, fontweight="bold")
        axes[row_cam][col].imshow(overlay)
        axes[row_cam][col].axis("off")
        if col==0:
            axes[row_cam][col].set_ylabel("Grad-CAM", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/kaggle/working/palewatch_gradcam.png", dpi=150, bbox_inches="tight")
plt.show()

all_confs_correct, all_confs_incorrect = [], []
sample_limit = 200
sample_pool = []
for cls_idx, paths in samples_by_class.items():
    for p in random.sample(paths, min(sample_limit//2, len(paths))):
        sample_pool.append((p, cls_idx))

model.eval()
with torch.no_grad():
    for img_path, true_label in sample_pool:
        pil = Image.open(img_path).convert("RGB")
        t = val_tf(pil).unsqueeze(0).to(DEVICE)
        prob = torch.softmax(model(t), dim=1)[0].cpu().numpy()
        pred = prob.argmax()
        conf = prob[pred]
        if pred==true_label:
            all_confs_correct.append(conf)
        else:
            all_confs_incorrect.append(conf)

fig2, ax = plt.subplots(figsize=(9,4))
ax.hist(all_confs_correct, bins=20, alpha=0.7, color="#55C788", label=f"Correct (n={len(all_confs_correct)})")
ax.hist(all_confs_incorrect, bins=20, alpha=0.7, color="#E05C5C", label=f"Incorrect (n={len(all_confs_incorrect)})")
ax.axvline(0.9, color="black", linestyle="--", lw=1.2, label="90% threshold")
ax.set_xlabel("Prediction Confidence")
ax.set_ylabel("Count")
ax.set_title("Model Confidence Distribution — Correct vs Incorrect Predictions")
ax.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/palewatch_confidence.png", dpi=150)
plt.show()
