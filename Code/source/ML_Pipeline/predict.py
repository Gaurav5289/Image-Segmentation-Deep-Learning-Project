# --- ✅ NEW predict.py ---
import cv2
import torch
import numpy as np
from ML_Pipeline.network import UNetPP
from argparse import ArgumentParser
from albumentations import Resize, Compose
from albumentations.augmentations import transforms
from albumentations.pytorch import ToTensorV2 # <-- Import ToTensorV2

# (Remove the old val_transform, it will be passed in)

def image_loader(image_path, transform):
    """
    Loads an image, applies transforms, and returns a tensor.
    """
    image = cv2.imread(image_path) # <-- Use cv2.imread
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # <-- Convert to RGB

    # Apply all transforms (Resize, Normalize, ToTensorV2)
    augmented = transform(image=image)
    image = augmented["image"]
    
    return image # <-- This is now a tensor, no more processing needed