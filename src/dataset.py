"""Data loading, stratified 70/15/15 split, and augmentation pipeline."""
import json
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE = 224
SPLIT_FILE = "data/processed/splits.json"


def list_images(raw_dir="data/raw"):
    raw = Path(raw_dir)
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in sorted((raw / c).rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                paths.append(str(p))
                labels.append(i)
    if not paths:
        raise FileNotFoundError(f"No images found under {raw_dir}/<class>/")
    return paths, labels, classes


def make_splits(raw_dir="data/raw", out_json=SPLIT_FILE, seed=42):
    """Stratified Train 70 / Val 15 / Test 15. Saved so both approaches share the same split."""
    paths, labels, classes = list_images(raw_dir)
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    out = {"classes": classes,
           "train": {"paths": p_tr, "labels": y_tr},
           "val": {"paths": p_va, "labels": y_va},
           "test": {"paths": p_te, "labels": y_te}}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w"))
    return out


def load_splits(raw_dir="data/raw", out_json=SPLIT_FILE):
    if Path(out_json).exists():
        return json.load(open(out_json))
    return make_splits(raw_dir, out_json)


def train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class AvocadoDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
