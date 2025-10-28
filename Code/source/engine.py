import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# --- 1. CORRECT ALBUMENTATIONS IMPORTS ---
from albumentations import (
    Resize, Compose, OneOf, RandomRotate90, Flip,
    HueSaturationValue, RandomBrightnessContrast # <-- Corrected
)
from albumentations.pytorch import ToTensorV2

# --- 2. Custom Imports (No Change) ---
from ML_Pipeline.utils import AverageMeter, iou_score
from ML_Pipeline.network import UNetPP
from ML_Pipeline.dataset import DataSet
from ML_Pipeline.train import train, validate
# (Make sure predict.py is fixed as we discussed)
from ML_Pipeline.predict import image_loader 

# ====================================================
# 📁 CORRECTED Config File Loading
# ====================================================

# Get the project root, NOT the current file's directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Code", "source")

config_path = os.path.join(SOURCE_DIR, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# --- Resolve all paths from config ---
extn = config["extn"]
epochs = config["epochs"]
# Use PROJECT_ROOT to build the correct absolute path
log_path = os.path.join(PROJECT_ROOT, config["log_path"])
mask_path = os.path.join(PROJECT_ROOT, config["mask_path"])
image_path = os.path.join(PROJECT_ROOT, config["image_path"])
model_path = os.path.join(PROJECT_ROOT, config["model_path"])
output_path = os.path.join(PROJECT_ROOT, config["output_path"])
patience = config.get("early_stopping_patience", 15)
batch_size = config.get("batch_size", 4)

# ====================================================
# 🧠 Data Preparation (Corrected Transforms)
# ====================================================

log = OrderedDict([
    ('epoch', []), ('loss', []), ('iou', []),
    ('val_loss', []), ('val_iou', []),
])
best_iou = 0
trigger = 0

# --- 🎯 ADDED BACK: Get all image IDs ---
img_ids = glob(os.path.join(image_path, f"*{extn}"))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

if not img_ids:
    raise ValueError(f"No images found in {image_path} with extension {extn}")

train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=42)
print(f"Data loaded: {len(train_img_ids)} train, {len(val_img_ids)} val.")


# --- 3. CORRECTED TRANSFORMS ---
train_transform = Compose([
    RandomRotate90(),
    Flip(),
    OneOf([
        HueSaturationValue(),
        RandomBrightnessContrast(), # <-- Use this one
    ], p=1),
    Resize(256, 256),
    Normalize(),
    ToTensorV2(), # <-- Add this
])

val_transform = Compose([
    Resize(256, 256),
    Normalize(),
    ToTensorV2(), # <-- Add this
])

# --- 🎯 ADDED BACK: Create Datasets ---
train_dataset = DataSet(
    img_ids=train_img_ids,
    image_path=image_path,
    mask_path=mask_path,
    img_ext=extn,
    mask_ext=extn,
    transform=train_transform,
)

val_dataset = DataSet(
    img_ids=val_img_ids,
    image_path=image_path,
    mask_path=mask_path,
    img_ext=extn,
    mask_ext=extn,
    transform=val_transform,
)

# --- 🎯 ADDED BACK: Create DataLoaders ---
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

# ====================================================
# 🧠 Model, Loss, Optimizer
# ====================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNetPP(num_classes=1, deep_supervision=True)
model = model.to(device) # <-- Send model to device

criterion = nn.BCEWithLogitsLoss()
params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params, lr=config.get("learning_rate", 1e-4), weight_decay=1e-4)

# ====================================================
# 🔁 Training Loop (Pass device)
# ====================================================

for epoch in range(epochs):
    print(f"\n🚀 Epoch [{epoch+1}/{epochs}]")

    # --- 4. PASS DEVICE ---
    train_log = train(True, train_loader, model, criterion, optimizer, device)
    val_log = validate(True, val_loader, model, criterion, device)

    # Print logs
    print(f"Epoch {epoch+1} | Train Loss: {train_log['loss']:.4f} | Train IoU: {train_log['iou']:.4f} | Val Loss: {val_log['loss']:.4f} | Val IoU: {val_log['iou']:.4f}")

    # Update log dictionary
    log['epoch'].append(epoch)
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])
    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])

    # Save log to CSV
    pd.DataFrame(log).to_csv(log_path, index=False)

    # Early stopping and model checkpointing
    trigger += 1
    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model Saved! IoU improved from {best_iou:.4f} to {val_log['iou']:.4f}")
        best_iou = val_log['iou']
        trigger = 0
    
    if trigger >= patience:
        print(f"⚠️ Early stopping! No improvement in {patience} epochs.")
        break

print("\n🎉 Training complete!")

    
# ====================================================
# 🧩 Prediction Section (Corrected Paths)
# ====================================================

print("\n--- Starting Prediction ---")

# --- 5. CORRECTED DEFAULT PATH ---
default_img_path = os.path.join(PROJECT_ROOT, "Code", "input", "PNG", "Original", "50.png")

parser = ArgumentParser()
parser.add_argument("--test_img", default=default_img_path, help="path to test image")
opt = parser.parse_args([]) 

im_width = config["im_width"]
im_height = config["im_height"]

# ... (Model loading is fine) ...
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- 6. CORRECTED PREDICTION LOGIC ---
image = image_loader(opt.test_img, val_transform) # Pass the transform
image = image.unsqueeze(0).to(device) # Add batch dim and send to device

with torch.no_grad():
    mask = model(image)
    
if isinstance(mask, list):
    mask = mask[-1] # Get final output

mask = torch.sigmoid(mask).squeeze().cpu().numpy()
mask_binary = (mask > 0.5).astype(np.uint8) * 255
mask_resized = cv2.resize(mask_binary, (im_width, im_height))

cv2.imwrite(output_path, mask_resized)
print(f"\n🩺 Prediction saved at: {output_path}")