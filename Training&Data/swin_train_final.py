"""
Swin Transformer — Final Integrated Training Script
monitor_v3 fully wired in. No manual integration needed.

Pre-run checklist (verify before running):
  1. Set CONFIG["data_dir"] to your dataset root
  2. Each class = one subfolder inside data_dir
  3. Run: python check_dataset.py  (generated below as __main__ block)
  4. Confirm EMA attribute after first model creation (printed automatically)
  5. Confirm class count printed matches your expectation
"""

import os
import sys
import json
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. REPRODUCIBILITY
# ─────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir":             "./data",   # root: expects class subfolders OR train/val subfolders
    "model_name":           "swin_small_patch4_window7_224",
    "image_size":           224,
    "batch_size":           16,
    "epochs":               60,
    "lr":                   7e-6,
    "weight_decay":         0.05,
    "warmup_epochs":        5,
    "min_lr":               1e-6,
    "label_smoothing":      0.1,
    "mixup_alpha":          0.03,
    "cutmix_alpha":         0.0,
    "mixup_start_epoch":    2,
    "n_folds":              5,
    "early_stop_patience":  10,
    "layer_decay":          0.75,
    "ema_decay":            0.99,
    "freeze_epochs":        0,
    "grad_accum_steps":     1,
    "num_workers":          0 if sys.platform == "win32" else 2,
    "device":               "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir":             "./checkpoints",
    "log_dir":              "./logs",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs(CONFIG["log_dir"],  exist_ok=True)

# These must match CONFIG — used by monitor
CONFIG_MIXUP_START   = CONFIG["mixup_start_epoch"]
CONFIG_WARMUP_EPOCHS = CONFIG["warmup_epochs"]

# ─────────────────────────────────────────────
# 2. MONITOR (inline, no import needed)
# ─────────────────────────────────────────────
class EpochLogger:
    def __init__(self, fold, class_names=None, save_dir="./logs"):
        os.makedirs(save_dir, exist_ok=True)
        self.fold        = fold
        self.class_names = class_names or []
        self.save_dir    = save_dir
        self.history     = []

    def log(self, epoch, train_loss, train_acc,
            val_loss, val_acc, val_f1,
            ema_val_loss=None, ema_val_acc=None,
            lr=None, pre_clip_norm=None, post_clip_norm=None,
            val_preds=None, val_labels=None):

        entry = {
            "epoch":          epoch,
            "train_loss":     round(train_loss,    5),
            "train_acc":      round(train_acc,     4),
            "val_loss":       round(val_loss,      5),
            "val_acc":        round(val_acc,       4),
            "val_f1":         round(val_f1,        4),
            "ema_val_loss":   round(ema_val_loss,  5) if ema_val_loss  is not None else None,
            "ema_val_acc":    round(ema_val_acc,   4) if ema_val_acc   is not None else None,
            "lr":             lr,
            "pre_clip_norm":  round(pre_clip_norm,  4) if pre_clip_norm  is not None else None,
            "post_clip_norm": round(post_clip_norm, 4) if post_clip_norm is not None else None,
        }
        self.history.append(entry)
        self._print(entry)
        self._save_json()
        self.diagnose(epoch)
        if val_preds is not None and val_labels is not None:
            if epoch % 5 == 0:
                self._check_per_class(val_preds, val_labels, epoch)

    def _print(self, e):
        ema_str  = f"  EMA {e['ema_val_acc']:.3f}"    if e["ema_val_acc"]   is not None else ""
        pre_str  = f"  GN {e['pre_clip_norm']:.3f}"   if e["pre_clip_norm"] is not None else ""
        post_str = f"→{e['post_clip_norm']:.3f}"       if e["post_clip_norm"]is not None else ""
        lr_str   = f"  LR {e['lr']:.2e}"               if e["lr"]            is not None else ""
        print(
            f"Ep {e['epoch']:3d} | "
            f"TrLoss {e['train_loss']:.4f} TrAcc {e['train_acc']:.3f} | "
            f"VaLoss {e['val_loss']:.4f} VaAcc {e['val_acc']:.3f} "
            f"F1 {e['val_f1']:.3f}"
            f"{ema_str}{pre_str}{post_str}{lr_str}"
        )

    def _save_json(self):
        path = os.path.join(self.save_dir, f"fold{self.fold}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def diagnose(self, epoch):
        h = self.history
        if len(h) < 2:
            return
        current  = h[-1]
        previous = h[-2]
        flags    = []

        # Overfitting — after epoch 8
        # Skip when train_acc=0 (mixup active — soft labels make hard-label acc meaningless)
        if epoch > 8 and current["train_acc"] > 0:
            gap = current["train_acc"] - current["val_acc"]
            if gap > 0.15:
                flags.append(
                    f"⚠️  OVERFIT GAP {gap:.3f}"
                    f"\n       👉 Try Mixup 0.2 or RandomErasing 0.2 first"
                )

        # Val loss spike
        abs_jump = current["val_loss"] - previous["val_loss"]
        if abs_jump > 0.05 and current["val_loss"] > previous["val_loss"] * 1.03:
            flags.append(
                f"⚠️  VAL LOSS SPIKE +{abs_jump:.4f}"
                f"\n       👉 Check LR or batch noise"
            )

        # Mixup transition window
        if CONFIG_MIXUP_START <= epoch <= CONFIG_MIXUP_START + 2:
            spike = current["val_loss"] - previous["val_loss"]
            if spike > 0.1:
                flags.append(
                    f"⚠️  MIXUP SPIKE +{spike:.4f} at epoch {epoch}"
                    f"\n       👉 Reduce mixup_alpha to 0.1–0.15"
                )
            elif epoch == CONFIG_MIXUP_START:
                flags.append(f"✅  Mixup started — spike: {spike:+.4f}")

        # Oscillation — direction flips + amplitude
        if len(h) >= 5:
            rv = [x["val_loss"] for x in h[-5:] if np.isfinite(x["val_loss"])]
            if len(rv) >= 3:
              dirs = [int(np.sign(rv[i] - rv[i-1])) for i in range(1, len(rv))]
            else:
              dirs = []
            if dirs.count(1) >= 2 and dirs.count(-1) >= 2 and abs(rv[-1] - rv[0]) > 0.05:
                flags.append(
                    f"⚠️  VAL LOSS OSCILLATING ({dirs.count(1)}↑ {dirs.count(-1)}↓)"
                    f"\n       👉 LR too high — check scheduler"
                )

        # Plateau
        if len(h) >= 10:
            last5 = [x["val_loss"] for x in h[-5:] if np.isfinite(x["val_loss"])]
            if len(last5) == 5 and max(last5) - min(last5) < 0.01:
                flags.append(
                    f"⚠️  PLATEAU (range {max(last5)-min(last5):.5f} over 5 epochs)"
                    f"\n       👉 Consider LR reduction or augmentation change"
                )

        # EMA
        if current["ema_val_acc"] is not None and epoch >= 10:
            diff = current["ema_val_acc"] - current["val_acc"]
            if diff < -0.01:
                flags.append(f"⚠️  EMA WORSE than raw by {abs(diff):.3f}\n       👉 Try decay 0.997")
            elif abs(diff) < 0.001:
                flags.append(f"⚠️  EMA IDENTICAL to raw\n       👉 Try decay 0.992")
            else:
                flags.append(f"✅  EMA gap: {diff:+.3f}")

            ema_recent = [x["ema_val_acc"] for x in h[-4:] if x["ema_val_acc"] is not None]
            if len(ema_recent) == 4 and ema_recent[-1] < min(ema_recent[:-1]):
                flags.append(
                    f"⚠️  EMA DECLINING 4-epoch ({ema_recent[0]:.3f}→{ema_recent[-1]:.3f})"
                    f"\n       👉 Possible stagnation"
                )

        # Grad norm — adaptive
        pre = current.get("pre_clip_norm")
        post = current.get("post_clip_norm")
        if pre is not None:
            recent_norms = [x["pre_clip_norm"] for x in h[-5:] if x.get("pre_clip_norm")]
            if len(recent_norms) >= 3:
                mean_gn = np.mean(recent_norms)
                if pre > mean_gn * 3:
                    flags.append(f"⚠️  GRAD SPIKE {pre:.3f} (3x mean {mean_gn:.3f})\n       👉 Check clip threshold")
                elif pre < mean_gn * 0.1:
                    flags.append(f"⚠️  GRAD DROP {pre:.4f} (0.1x mean)\n       👉 Vanishing — check frozen layers / LR")
            if post is not None and pre > 0:
                if (post / pre) < 0.5:
                    flags.append(f"⚠️  HEAVY CLIPPING {pre:.3f}→{post:.3f}\n       👉 Gradients clipped >50%")

        # LR post-warmup
        if current["lr"] is not None and epoch > CONFIG_WARMUP_EPOCHS:
            if current["lr"] > 2e-4:
                flags.append(f"⚠️  LR {current['lr']:.2e} high post-warmup")

        # Early stall
        if epoch == 10:
            best_val = min(x["val_loss"] for x in h)
            if best_val >= h[0]["val_loss"] * 0.98:
                flags.append(
                    f"🚨  NO IMPROVEMENT after 10 epochs"
                    f"\n       👉 Stop. Check data, labels, LR warmup."
                )

        if flags:
            print("  DIAGNOSTICS:")
            for fl in flags:
                for line in fl.split("\n"):
                    print(f"    {line}")

    def _check_per_class(self, preds, labels, epoch):
        if not self.class_names:
            return
        report = classification_report(
            labels, preds, target_names=self.class_names,
            output_dict=True, zero_division=0
        )
        class_f1 = {k: v["f1-score"] for k, v in report.items() if k in self.class_names}
        if not class_f1:
            return
        worst       = min(class_f1, key=class_f1.get)
        worst_f1    = class_f1[worst]
        mean_f1     = np.mean(list(class_f1.values()))
        print(f"\n  PER-CLASS (epoch {epoch}):")
        for cls, f1 in sorted(class_f1.items(), key=lambda x: x[1]):
            bar  = "█" * int(f1 * 20)
            warn = " ⚠️" if f1 < 0.5 else ""
            print(f"    {cls:<25} F1={f1:.3f} |{bar}{warn}")
        if worst_f1 < 0.5:
            print(f"\n    🚨 WORST: '{worst}' F1={worst_f1:.3f} (mean={mean_f1:.3f})")
            print(f"       👉 Targeted augmentation or higher class weight")
        with open(os.path.join(self.save_dir, f"fold{self.fold}_perclass_ep{epoch}.json"), "w") as f:
            json.dump(class_f1, f, indent=2)

    def plot(self):
        h      = self.history
        epochs = [e["epoch"]      for e in h]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(f"Fold {self.fold} — Training Monitor", fontsize=13)

        axes[0].plot(epochs, [e["train_loss"] for e in h], label="Train", color="#2196F3")
        axes[0].plot(epochs, [e["val_loss"]   for e in h], label="Val",   color="#F44336")
        ema_l = [(e["epoch"], e["ema_val_loss"]) for e in h if e["ema_val_loss"] is not None]
        if ema_l:
            ex, ey = zip(*ema_l); axes[0].plot(ex, ey, "--", label="EMA", color="#FF9800")
        axes[0].axvline(CONFIG_MIXUP_START, color="gray", linestyle=":", alpha=0.6)
        axes[0].set_title("Loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, [e["train_acc"] for e in h], label="Train", color="#2196F3")
        axes[1].plot(epochs, [e["val_acc"]   for e in h], label="Val",   color="#F44336")
        ema_a = [(e["epoch"], e["ema_val_acc"]) for e in h if e["ema_val_acc"] is not None]
        if ema_a:
            ex, ey = zip(*ema_a); axes[1].plot(ex, ey, "--", label="EMA", color="#FF9800")
        axes[1].set_title("Accuracy"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        pre_d  = [(e["epoch"], e["pre_clip_norm"])  for e in h if e.get("pre_clip_norm")]
        post_d = [(e["epoch"], e["post_clip_norm"]) for e in h if e.get("post_clip_norm")]
        if pre_d:
            px, py = zip(*pre_d);  axes[2].plot(px, py, label="Pre-clip",  color="#9C27B0")
        if post_d:
            qx, qy = zip(*post_d); axes[2].plot(qx, qy, "--", label="Post-clip", color="#E91E63")
        axes[2].set_title("Grad Norm"); axes[2].set_yscale("log")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"fold{self.fold}_curves.png")
        plt.savefig(path, dpi=120); plt.close()
        print(f"  Plot → {path}")

    def summary(self):
        h       = self.history
        best_ep = min(h, key=lambda x: x["val_loss"])
        last    = h[-1]
        print(f"\n{'─'*52}\n  FOLD {self.fold} SUMMARY\n{'─'*52}")
        print(f"  Best val loss : {best_ep['val_loss']:.4f} (ep {best_ep['epoch']}, acc={best_ep['val_acc']:.4f})")
        if last["train_acc"] > 0:
            gap = last["train_acc"] - last["val_acc"]
            print(f"  Train-val gap : {gap:.3f}")
            if   gap > 0.15: print("  ⚠️  Overfitting.")
            elif gap < 0.03: print("  ✅  Tight gap.")
            else:            print("  ✅  Reasonable.")
        else:
            print("  Train-val gap : N/A (mixup active — hard-label train acc not tracked)")
        print(f"  Epochs run    : {len(h)}")

# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
class RawImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        return self.loader(path), target

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

# ─────────────────────────────────────────────
# 4. AUGMENTATION
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(CONFIG["image_size"], scale=(0.8, 1.0)),  # local feature learning
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

# ─────────────────────────────────────────────
# 5. LAYER-WISE LR DECAY
# ─────────────────────────────────────────────
def build_layerwise_params(model, base_lr, layer_decay, weight_decay):
    num_stages   = len(model.layers)
    layer_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = ("bias" in name) or ("norm" in name)
        if "patch_embed" in name:
            layer_id = 0
        elif "layers." in name:
            try:    layer_id = int(name.split("layers.")[1].split(".")[0]) + 1
            except: layer_id = num_stages
        else:
            layer_id = num_stages
        layer_groups.setdefault((layer_id, no_decay), []).append(param)

    param_groups = []
    for (layer_id, no_decay), params in layer_groups.items():
        lr_scale = layer_decay ** (num_stages - layer_id)
        param_groups.append({
            "params":       params,
            "lr":           base_lr * lr_scale,
            "weight_decay": 0.0 if no_decay else weight_decay,
        })
    return param_groups

# ─────────────────────────────────────────────
# 6. SCHEDULER
# ─────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        e = self.current_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale    = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(base_lr * scale, self.min_lr)
        self.current_epoch += 1

# ─────────────────────────────────────────────
# 7. GRAD NORM
# ─────────────────────────────────────────────
def compute_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

# ─────────────────────────────────────────────
# 8. LOSS FUNCTIONS
# ─────────────────────────────────────────────
# During mixup: plain SoftTargetCrossEntropy — no class weights.
# Class weights during mixup over-amplify gradient scale and dilute rare-class identity.
# Mixup provides implicit balance by interpolating labels across classes.
# During hard labels: weighted CrossEntropyLoss handles class imbalance.
soft_criterion = SoftTargetCrossEntropy()

# ─────────────────────────────────────────────
# 9. TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, hard_criterion,
                    mixup_fn, scaler, device, epoch, model_ema=None):
    model.train()
    use_mixup      = (epoch >= CONFIG["mixup_start_epoch"])
    # During mixup: plain SoftTargetCrossEntropy (no class weights)
    # During hard labels: weighted CrossEntropyLoss for class imbalance
    total_loss     = 0.0
    correct        = 0
    total          = 0
    pre_clip_accum = 0.0
    post_clip_accum= 0.0
    steps          = len(loader)
    optimizer.zero_grad()
    nan_detected = False

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mixup:
            imgs, mixed_targets = mixup_fn(imgs, labels)
            with torch.cuda.amp.autocast(enabled=False):  # FP32: stability > speed on small dataset
                outputs = model(imgs)
                # Plain soft CE — no class weights during mixup (see loss function section)
                loss    = soft_criterion(outputs, mixed_targets) / CONFIG["grad_accum_steps"]
        else:
            with torch.cuda.amp.autocast(enabled=False):  # FP32: stability > speed on small dataset
                outputs = model(imgs)
                loss    = hard_criterion(outputs, labels) / CONFIG["grad_accum_steps"]

        if not torch.isfinite(loss):
            print(f"  🚨  NaN/Inf loss at step {step} — resetting optimizer state and stopping epoch")
            # Reset optimizer momentum buffers — corrupted state poisons all future steps
            optimizer.state.clear()
            optimizer.zero_grad()
            # DO NOT call scaler.update() here — no inf check was registered (backward not called)
            nan_detected = True
            break
        scaler.scale(loss).backward()

        last_step = (step + 1 == steps)
        if (step + 1) % CONFIG["grad_accum_steps"] == 0 or last_step:
            scaler.unscale_(optimizer)

            # Grad norm — BEFORE clip
            pre_clip  = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # Grad norm — AFTER clip
            post_clip = compute_grad_norm(model)

            pre_clip_accum  += pre_clip
            post_clip_accum += post_clip

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA update AFTER optimizer step only — not every forward pass
            if model_ema is not None:
                model_ema.update(model)

        total_loss += loss.item() * CONFIG["grad_accum_steps"]
        if not use_mixup:  # hard labels only — mixup accuracy is meaningless
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)

    update_steps = max(1, steps // CONFIG["grad_accum_steps"])
    train_acc = correct / total if total > 0 else 0.0  # 0 when all steps used mixup
    return (
        total_loss / steps,
        train_acc,
        pre_clip_accum  / update_steps,
        post_clip_accum / update_steps,
        nan_detected,
    )

# ─────────────────────────────────────────────
# 10. EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=False):  # FP32: match training precision
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / len(loader), acc, f1, all_preds, all_labels

def evaluate_tta(model, pil_subset, criterion, device, tta_transforms):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for idx in range(len(pil_subset)):
            img_pil, label = pil_subset[idx]
            label_t        = torch.tensor([label], device=device)
            logits_sum     = None
            for tfm in tta_transforms:
                img_t = tfm(img_pil).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=False):  # FP32: match training precision
                    logits = model(img_t)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            avg = logits_sum / len(tta_transforms)
            avg     = avg.float()        # ensure FP32 — AMP may leave outputs as FP16
            label_t = label_t.long()
            total_loss += criterion(avg, label_t).item()
            all_preds.append(avg.argmax(1).item())
            all_labels.append(label)
    n   = len(pil_subset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / n, acc, f1, all_preds, all_labels

# ─────────────────────────────────────────────
# 11. DATASET LOAD + SANITY
# ─────────────────────────────────────────────
print("=" * 55)
print("  DATASET SANITY CHECK")
print("=" * 55)

def load_dataset(data_dir):
    """
    Handles two structures:
      A) data/<class>/...          → single ImageFolder
      B) data/train/<class>/...
         data/val/<class>/...      → merge both, re-split via StratifiedKFold
    """
    import os
    from torch.utils.data import ConcatDataset

    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")

    if os.path.isdir(train_path) and os.path.isdir(val_path):
        print("  Detected train/val subfolder structure → merging for CV")
        ds_train = RawImageFolder(root=train_path)
        ds_val   = RawImageFolder(root=val_path)

        # Verify both have identical classes — name, order, and count
        train_classes = sorted(ds_train.classes)
        val_classes   = sorted(ds_val.classes)
        if train_classes != val_classes:
            only_train = set(ds_train.classes) - set(ds_val.classes)
            only_val   = set(ds_val.classes)   - set(ds_train.classes)
            raise RuntimeError(
                f"Class mismatch between train and val folders.\n"
                f"  Only in train : {only_train or 'none'}\n"
                f"  Only in val   : {only_val   or 'none'}\n"
                f"  Fix: folder names must be identical (case-sensitive)"
            )
        # Verify class-to-index mapping is consistent
        if ds_train.class_to_idx != ds_val.class_to_idx:
            raise RuntimeError(
                f"class_to_idx mismatch — same names but different order.\n"
                f"  train: {ds_train.class_to_idx}\n"
                f"  val:   {ds_val.class_to_idx}\n"
                f"  This should not happen if both use ImageFolder defaults."
            )

        # Merge samples and targets manually (ConcatDataset loses .targets)
        combined = RawImageFolder(root=train_path)  # use as base for class info
        combined.samples = ds_train.samples + ds_val.samples
        combined.targets = ds_train.targets + ds_val.targets
        combined.imgs    = combined.samples
        return combined

    else:
        print("  Detected flat class structure → loading directly")
        return RawImageFolder(root=data_dir)

full_dataset = load_dataset(CONFIG["data_dir"])
targets      = np.array(full_dataset.targets)
classes      = full_dataset.classes
num_classes  = len(classes)
CONFIG["num_classes"] = num_classes

class_counts = np.bincount(targets)
print(f"Total images : {len(full_dataset)}")
print(f"Classes ({num_classes}):")
for i, c in enumerate(classes):
    print(f"  {c:<30} {class_counts[i]:4d} images")

empty = [c for i, c in enumerate(classes) if class_counts[i] == 0]
if empty:
    raise RuntimeError(f"Empty class folders detected: {empty}")

# Class weights
class_weights        = 1.0 / class_counts           # stronger: full inverse count
class_weights        = class_weights / class_weights.sum() * len(class_weights)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(CONFIG["device"])

print(f"\nDevice : {CONFIG['device']}")
print("=" * 55)

# ─────────────────────────────────────────────
# 12. 5-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────
skf          = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"\n{'='*55}\n  FOLD {fold+1} / {CONFIG['n_folds']}\n{'='*55}")

    train_ds = TransformSubset(Subset(full_dataset, train_idx), train_transform)
    val_ds   = TransformSubset(Subset(full_dataset, val_idx),   val_transform)
    val_pil  = Subset(full_dataset, val_idx)

    _pin = CONFIG["device"] == "cuda"
    _nw  = CONFIG["num_workers"]

    # WeightedRandomSampler — oversample rare classes so each batch
    # has proportional representation. Uses fold train indices only.
    # No normalization needed — sampler only uses relative weights.
    train_labels   = targets[train_idx]
    sample_weights = np.array([1.0 / class_counts[t] for t in train_labels], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              sampler=sampler, num_workers=_nw, pin_memory=_pin,
                              drop_last=True)  # prevents odd last batch crashing Mixup
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=_nw, pin_memory=_pin)

    # Model
    model = timm.create_model(
        CONFIG["model_name"], pretrained=True,
        num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.1,
    ).to(CONFIG["device"])

    # EMA attribute check (printed once on fold 1)
    model_ema = ModelEmaV2(model, decay=CONFIG["ema_decay"])
    if fold == 0:
        has_module = hasattr(model_ema, "module")
        print(f"  EMA attribute check → .module exists: {has_module}")
        if not has_module:
            raise AttributeError(
                "ModelEmaV2 has no .module attribute. "
                "Check your timm version — may need .ema instead."
            )

    # Full model trains from epoch 1 — no freezing.
    # layer-wise LR decay handles the "lower LR for early layers" concern.

    param_groups = build_layerwise_params(
        model, CONFIG["lr"], CONFIG["layer_decay"], CONFIG["weight_decay"]
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineScheduler(
        optimizer, CONFIG["warmup_epochs"], CONFIG["epochs"], CONFIG["min_lr"]
    )

    # Mixup probability ramp: 0.3 → 0.6 (ep 8) → 1.0 (ep 15)
    # Gradual ramp avoids regularization shock in early epochs
    def make_mixup_fn(prob):
        return Mixup(
            mixup_alpha=CONFIG["mixup_alpha"], cutmix_alpha=CONFIG["cutmix_alpha"],
            prob=prob, switch_prob=0.5, mode="batch",
            label_smoothing=CONFIG["label_smoothing"], num_classes=num_classes,
        )
    mixup_fn = make_mixup_fn(0.3)   # start conservative
    # Loss functions — no class weights in loss: sampler already oversamples rare classes.
    # Adding weights on top creates double amplification that makes loss harder to optimize.
    val_criterion  = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    hard_criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    scaler           = torch.cuda.amp.GradScaler(enabled=CONFIG["device"]=="cuda")
    logger           = EpochLogger(fold=fold+1, class_names=classes, save_dir=CONFIG["log_dir"])
    best_val_loss    = float("inf")
    best_ckpt_path   = None          # global best — never deleted
    patience_counter = 0
    top_k            = []            # (ema_loss, path) — top-3 recent, cleaned up

    for epoch in range(1, CONFIG["epochs"] + 1):

        # No freeze/unfreeze logic — full model trains from epoch 1.
        # Mixup ramps from prob=0.3 (start) to 1.0 at epoch 15 for smooth regularization.
        if epoch == 15:
            mixup_fn = make_mixup_fn(0.7)
            print(f"  [Epoch {epoch}] Mixup → prob=0.7 (full regularization capped — preserves rare class signal)")
        elif epoch == 8:
            mixup_fn = make_mixup_fn(0.6)
            print(f"  [Epoch {epoch}] Mixup → prob=0.6")

        train_loss, train_acc, pre_norm, post_norm, nan_hit = train_one_epoch(
            model, train_loader, optimizer, hard_criterion,
            mixup_fn, scaler, CONFIG["device"], epoch, model_ema
        )
        if nan_hit:
            print(f"  🚨  NaN detected in epoch {epoch} — aborting fold {fold+1}. Restart required.")
            break
        scheduler.step()

        # Evaluate: EMA model
        val_loss, val_acc, val_f1, val_preds, val_labels_out = evaluate(
            model_ema.module, val_loader, val_criterion, CONFIG["device"]
        )
        # Also get raw EMA metrics for logger
        ema_loss = val_loss
        ema_acc  = val_acc

        # Raw model eval for comparison
        raw_loss, raw_acc, raw_f1, _, _ = evaluate(
            model, val_loader, val_criterion, CONFIG["device"]
        )

        current_lr = optimizer.param_groups[-1]["lr"]

        logger.log(
            epoch, train_loss, train_acc,
            raw_loss, raw_acc, raw_f1,          # raw model as "val" baseline
            ema_val_loss=ema_loss,
            ema_val_acc=ema_acc,
            lr=current_lr,
            pre_clip_norm=pre_norm,
            post_clip_norm=post_norm,
            val_preds=val_preds,
            val_labels=val_labels_out,
        )

        # ── Checkpoint (EMA weights) ──────────────────────────────────────
        ckpt = os.path.join(CONFIG["save_dir"], f"fold{fold+1}_ep{epoch}_vl{ema_loss:.4f}.pt")
        torch.save(model_ema.module.state_dict(), ckpt)

        # 1. Global best — copied to a fixed path, NEVER deleted or overwritten
        #    by the top-k cleanup. TTA and early stopping both use this.
        if ema_loss < best_val_loss:
            best_val_loss    = ema_loss
            patience_counter = 0
            best_ckpt_path   = os.path.join(CONFIG["save_dir"], f"fold{fold+1}_BEST.pt")
            shutil.copy(ckpt, best_ckpt_path)
            print(f"  🔥  New best → {best_ckpt_path}  (loss={ema_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

        # 2. Top-K rolling window — keeps 3 recent checkpoints for flexibility.
        #    Never deletes the global best even if it falls out of the window.
        top_k.append((ema_loss, ckpt))
        top_k.sort(key=lambda x: x[0])
        if len(top_k) > 3:
            _, worst_path = top_k.pop()   # remove worst (highest loss)
            if worst_path != best_ckpt_path and os.path.exists(worst_path):
                os.remove(worst_path)

    # Final TTA eval — load global best EMA checkpoint for this fold
    model_ema.module.load_state_dict(torch.load(best_ckpt_path))
    model_ema.module.to(CONFIG["device"])
    print(f"\n  TTA evaluation — fold {fold+1}...")
    tta_loss, tta_acc, tta_f1, tta_preds, tta_labels = evaluate_tta(
        model_ema.module, val_pil, val_criterion,
        CONFIG["device"], val_tta_transforms
    )
    print(f"\nFold {fold+1} TTA → Acc: {tta_acc:.4f}  F1: {tta_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(tta_labels, tta_preds, target_names=classes, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(tta_labels, tta_preds))

    logger.plot()
    logger.summary()
    fold_results.append({"fold": fold+1, "acc": tta_acc, "f1": tta_f1})

# ─────────────────────────────────────────────
# 13. FINAL CV SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*55}\n  CROSS-VALIDATION SUMMARY\n{'='*55}")
accs = [r["acc"] for r in fold_results]
f1s  = [r["f1"]  for r in fold_results]
for r in fold_results:
    print(f"  Fold {r['fold']}: Acc={r['acc']:.4f}  F1={r['f1']:.4f}")
print(f"\n  Mean Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"  Mean F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
