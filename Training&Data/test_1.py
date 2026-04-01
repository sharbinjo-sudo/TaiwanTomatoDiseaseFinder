import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import os
import random

# CONFIG
# MODEL_PATH points to the saved state dict file produced by the training script.
# This should be the best EMA checkpoint from cross-validation.
# DATA_PATH should be the root folder containing one subfolder per class.
# CLASS_NAMES must be listed in the exact same order used during training.
# The index of each name here must match the class-to-index mapping that
# ImageFolder assigned when the training dataset was loaded. A mismatch
# silently produces wrong label strings even when the model predicts correctly.
MODEL_PATH = "best_model.pth"
DATA_PATH = "data/train"

CLASS_NAMES = [
    "Bacterial spot",
    "Black mold",
    "Gray spot",
    "Late blight",
    "health",
    "powdery mildew"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL
# The architecture must match exactly what was used during training, including
# the number of output classes. pretrained=False is correct here because the
# weights will be loaded from the local checkpoint, not downloaded from the
# timm model hub.
model = timm.create_model(
    "swin_small_patch4_window7_224",
    pretrained=False,
    num_classes=len(CLASS_NAMES)
)

# load_state_dict loads only the parameter tensors, not the optimizer state or
# scheduler state, which is all that is needed for inference. map_location
# ensures the checkpoint is remapped to the correct device even if it was
# saved on a GPU and is now being loaded on a CPU machine.
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

# eval mode disables dropout and sets batch normalization layers to use their
# running statistics rather than batch statistics. Forgetting this call causes
# inconsistent predictions because dropout randomly zeros activations at
# inference time, and batch norm behaves differently in train mode.
model.eval()

print("Model loaded")

# TRANSFORM
# The transform here must match the validation transform used during training,
# not the training augmentation pipeline. The model was evaluated against
# clean resized images during cross-validation, so applying augmentation at
# test time would produce a distribution mismatch.
# Note: ImageNet normalization (mean/std) is absent here. If the training
# script's val_transform included Normalize([0.485, 0.456, 0.406], [...]),
# this transform should include it too for consistent predictions.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# FIND VALID CLASS FOLDERS
# Scans DATA_PATH for subdirectories only, ignoring any loose files at the
# root level. This mirrors how ImageFolder discovers classes during training.
class_folders = [
    f for f in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, f))
]

if len(class_folders) == 0:
    raise ValueError("No class folders found. Check DATA_PATH")

# PICK RANDOM CLASS
# Selects a random class folder so the test script exercises different classes
# across multiple runs rather than always testing the same one.
class_folder = random.choice(class_folders)
img_folder = os.path.join(DATA_PATH, class_folder)

# GET IMAGES FROM THAT CLASS
# Filters to supported image formats only. Files with unsupported extensions
# would cause Image.open to raise an error further down.
image_files = [
    f for f in os.listdir(img_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(image_files) == 0:
    raise ValueError(f"No images found in {img_folder}")

# PICK RANDOM IMAGE
img_name = random.choice(image_files)
img_path = os.path.join(img_folder, img_name)

print("\nTesting image:", img_path)
print("Actual class:", class_folder)

# LOAD IMAGE
# convert("RGB") ensures the image always has 3 channels. PNG files can have
# 4 channels (RGBA) and grayscale images have 1 channel. The Swin Transformer
# expects exactly 3 channels, so this conversion handles both edge cases.
img = Image.open(img_path).convert("RGB")

# PREPROCESS
# unsqueeze(0) adds a batch dimension, changing the tensor shape from
# (3, 224, 224) to (1, 3, 224, 224). The model expects a batch as input,
# not a single image tensor.
x = transform(img).unsqueeze(0).to(DEVICE)

# INFERENCE
# torch.no_grad() disables gradient computation for the forward pass.
# This reduces memory usage and speeds up inference because PyTorch does not
# need to build a computation graph when no backward pass will follow.
# softmax converts the raw output logits into a probability distribution
# that sums to 1 across all classes. torch.max then returns the highest
# probability and its corresponding class index in a single call.
with torch.no_grad():
    out = model(x)
    probs = torch.softmax(out, dim=1)
    conf, pred = torch.max(probs, 1)

# pred.item() converts the single-element tensor to a plain Python int so it
# can be used as a list index into CLASS_NAMES.
pred_class = CLASS_NAMES[pred.item()]
confidence = conf.item()

print("\nPredicted:", pred_class)
print("Confidence:", round(confidence, 4))
