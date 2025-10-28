import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, img_ids, image_path, mask_path, img_ext, mask_ext, transform=None):
        self.img_ids = img_ids
        self.image_path = image_path
        self.mask_path = mask_path
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __getitem__(self, i):
        img_id = self.img_ids[i]

        img_file = os.path.join(self.image_path, img_id + self.img_ext)
        mask_file = os.path.join(self.mask_path, img_id + self.mask_ext)

        # ✅ Read safely
        image = cv2.imread(img_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"❌ Image not found: {img_file}")  
        if mask is None:
            raise FileNotFoundError(f"❌ Mask not found: {mask_file}")

        mask = mask[..., None]  # expand channel

       # ... (previous code) ...
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # # ---------------- PROBLEM IS HERE ----------------
        # image = image.transpose(2, 0, 1)  # <-- DELETE THIS
        # mask = mask.transpose(2, 0, 1)    # <-- DELETE THIS
        # # -------------------------------------------------

        return image, mask, img_id

    def __len__(self):
        return len(self.img_ids)
