"""
PolypGen Colonoscopy Dataset Loading & Endoscopy Data Augmentation Pipeline.
Natively compatible with the official Synapse PolypGen Benchmark (syn26376615):
https://www.synapse.org/Synapse:syn26376615/wiki/613312

Supports:
- Multi-center data: data_C1, data_C2, data_C3, data_C4, data_C5, data_C6
- Sequence folders (seq_*) and still images
- Automatic exclusion of ground-truth segmentation masks (masks/, *_mask.*)
- 70/15/15 train/val/test splits with dynamic randomized seeds
"""

import os
import sys
import random
import secrets
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["0_NO_POLYP", "1_POLYP"]
CLASS_DESCRIPTIONS = {
    "0_NO_POLYP": "Normal Mucosa / Negative Frame",
    "1_POLYP": "Polyp Present / Positive Frame"
}


def setup_random_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = secrets.randbelow(900000) + 100000
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[Random-Seed] Active experiment seed: {seed}")
    return seed


class PolypDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: np.ndarray, transform=None):
        self.image_paths = image_paths
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_endoscopy_transforms(
    img_size: int = 224,
    is_training: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def is_mask_file_or_folder(filepath: Path) -> bool:
    """
    Checks if a file or directory represents segmentation mask / annotation metadata.
    """
    path_str = str(filepath).lower().replace("\\", "/")
    filename = filepath.name.lower()

    # Mask directory keywords
    mask_dirs = ["/masks/", "/mask/", "/ground_truth/", "/labels/", "/annotations/"]
    if any(md in path_str for md in mask_dirs):
        return True

    # Mask filename keywords
    mask_suffixes = ["_mask", "-mask", "_label", "_gt", "-gt", "_seg"]
    for sfx in mask_suffixes:
        if sfx in filename:
            return True

    # Non-image files
    non_img_exts = (".txt", ".json", ".csv", ".mat", ".xml", ".nii", ".gz", ".zip")
    if filename.endswith(non_img_exts):
        return True

    return False


def is_valid_colonoscopy_image(filepath: Path) -> bool:
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not filepath.name.lower().endswith(valid_exts):
        return False
    return not is_mask_file_or_folder(filepath)


def create_synthetic_demo_dataset(data_dir: Path, num_samples_per_class: int = 100):
    print(f"[Dataset] Creating PolypGen demo dataset in {data_dir} ({num_samples_per_class*2} frames)...")
    for idx, cname in enumerate(CLASS_NAMES):
        folder = data_dir / cname
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples_per_class):
            img_path = folder / f"frame_{i:04d}.jpg"
            if img_path.exists():
                continue
            
            base = Image.new("RGB", (300, 300), color=(
                random.randint(180, 230),
                random.randint(70, 110),
                random.randint(80, 120)
            ))
            draw = ImageDraw.Draw(base)

            for _ in range(15):
                x1, y1 = random.randint(0, 300), random.randint(0, 300)
                x2, y2 = x1 + random.randint(-40, 40), y1 + random.randint(-40, 40)
                draw.line([(x1, y1), (x2, y2)], fill=(random.randint(140, 190), 40, 50), width=random.randint(1, 3))

            if idx == 1:
                px, py = random.randint(80, 220), random.randint(80, 220)
                pr = random.randint(30, 60)
                draw.ellipse(
                    [(px - pr, py - pr), (px + pr, py + pr)],
                    fill=(random.randint(190, 240), random.randint(90, 140), random.randint(90, 140)),
                    outline=(150, 40, 40),
                    width=2
                )

            base = base.filter(ImageFilter.GaussianBlur(radius=1))
            base.save(img_path, quality=90)


def load_polyp_dataset_paths_and_labels(data_dir: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Recursively scans data_dir for both:
    1. Synapse PolypGen multi-center structure (data_C1 .. data_C6, negative_only, images, seq_*)
    2. Standard 2-folder structure (0_NO_POLYP vs 1_POLYP or negative vs positive)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = [], []
    center_counts = {}

    # Check for Synapse multi-center hierarchy (e.g. data_C1, data_C2 ... data_C6 or center_*)
    all_subdirs = [d for d in data_dir.rglob("*") if d.is_dir()]
    if not all_subdirs:
        all_subdirs = [d for d in data_dir.iterdir() if d.is_dir()]

    synapse_center_dirs = [
        d for d in all_subdirs 
        if any(marker in d.name.lower() for marker in ["data_c", "center_", "centre_", "c1", "c2", "c3", "c4", "c5", "c6"])
    ]

    neg_keywords = ["negative_only", "negative_samples", "negative_frames", "negative", "normal", "0_no_polyp", "no_polyp", "non_polyp"]
    pos_keywords = ["positive_frames", "positive_samples", "positive", "images", "polyp", "polyps", "1_polyp"]

    if synapse_center_dirs:
        print(f"[Dataset] Discovered {len(synapse_center_dirs)} Synapse multi-center folders in '{data_dir}'...")
        for center_dir in synapse_center_dirs:
            cname = center_dir.name
            for f in center_dir.rglob("*"):
                if f.is_file() and is_valid_colonoscopy_image(f):
                    path_str = str(f).lower().replace("\\", "/")
                    
                    # Classify as negative or positive based on path
                    if any(nk in path_str for nk in neg_keywords):
                        image_paths.append(str(f.resolve()))
                        labels.append(0)
                        center_counts[f"{cname}_Negative"] = center_counts.get(f"{cname}_Negative", 0) + 1
                    elif any(pk in path_str for pk in pos_keywords):
                        image_paths.append(str(f.resolve()))
                        labels.append(1)
                        center_counts[f"{cname}_Positive"] = center_counts.get(f"{cname}_Positive", 0) + 1
    else:
        # Standard folder layout check
        for f in data_dir.rglob("*"):
            if f.is_file() and is_valid_colonoscopy_image(f):
                path_str = str(f).lower().replace("\\", "/")
                if any(nk in path_str for nk in neg_keywords):
                    image_paths.append(str(f.resolve()))
                    labels.append(0)
                elif any(pk in path_str for pk in pos_keywords):
                    image_paths.append(str(f.resolve()))
                    labels.append(1)

    # If no data found, generate synthetic demo dataset
    if len(image_paths) == 0:
        print(f"[Dataset] No images found in '{data_dir}'. Generating demo dataset...")
        create_synthetic_demo_dataset(data_dir, num_samples_per_class=100)
        return load_polyp_dataset_paths_and_labels(data_dir)

    labels_arr = np.array(labels, dtype=np.int64)
    neg_count = int(np.sum(labels_arr == 0))
    pos_count = int(np.sum(labels_arr == 1))

    print(f"[Dataset] Successfully indexed {len(image_paths)} colonoscopy images (0_NO_POLYP: {neg_count}, 1_POLYP: {pos_count}).")
    if center_counts:
        print(f"[Dataset] Synapse Multi-Center Breakdown: {center_counts}")

    return image_paths, labels_arr, CLASS_NAMES


def create_polyp_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    seed: Optional[int] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    subsample: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int]:
    active_seed = setup_random_seed(seed)
    image_paths, labels, class_names = load_polyp_dataset_paths_and_labels(data_dir)

    if subsample is not None and subsample < len(image_paths):
        np.random.seed(active_seed)
        sub_idx = np.random.choice(len(image_paths), size=subsample, replace=False)
        image_paths = [image_paths[i] for i in sub_idx]
        labels = labels[sub_idx]

    holdout_ratio = val_split + test_split
    train_paths, holdout_paths, train_lbls, holdout_lbls = train_test_split(
        image_paths, labels, test_size=holdout_ratio, stratify=labels, random_state=active_seed
    )

    test_rel_ratio = test_split / holdout_ratio
    val_paths, test_paths, val_lbls, test_lbls = train_test_split(
        holdout_paths, holdout_lbls, test_size=test_rel_ratio, stratify=holdout_lbls, random_state=active_seed
    )

    print(f"[Dataset] Split sizes: Train={len(train_paths)} (70%), Val={len(val_paths)} (15%), Test={len(test_paths)} (15%)")

    train_tf = get_endoscopy_transforms(img_size=img_size, is_training=True, mean=mean, std=std)
    eval_tf = get_endoscopy_transforms(img_size=img_size, is_training=False, mean=mean, std=std)

    train_dataset = PolypDataset(train_paths, train_lbls, transform=train_tf)
    val_dataset = PolypDataset(val_paths, val_lbls, transform=eval_tf)
    test_dataset = PolypDataset(test_paths, test_lbls, transform=eval_tf)

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    return train_loader, val_loader, test_loader, class_names, active_seed


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_path = current_dir / "data"
    tr_loader, vl_loader, ts_loader, names, s = create_polyp_dataloaders(data_path, batch_size=16)
    for imgs, lbls in tr_loader:
        print(f"[Sanity Test] Batch tensor shape: {imgs.shape}, Labels: {lbls.shape}, Seed: {s}")
        break
