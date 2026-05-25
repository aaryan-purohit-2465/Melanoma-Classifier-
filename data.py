import os
import random

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2

    CV2_OK = True
except Exception:
    cv2 = None
    CV2_OK = False

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_OK = True
except Exception:
    torch = None
    DataLoader = None
    Dataset = object
    TORCH_OK = False


IMG_SIZE = 224
SMALL_SIZE = 32
CLASSES = ["thin", "intermediate", "thick"]
BINARY_CLASSES = ["thin", "thick"]


def make_dirs():
    a = [
        "dataset/train/thin",
        "dataset/train/intermediate",
        "dataset/train/thick",
        "dataset/test/thin",
        "dataset/test/intermediate",
        "dataset/test/thick",
        "models",
        "results",
        "uploads",
    ]
    for b in a:
        os.makedirs(b, exist_ok=True)


def draw_dummy_image(a):
    x = Image.new("RGB", (256, 256), (25, 25, 25))
    y = ImageDraw.Draw(x)
    z = {
        "thin": (60, 170, 90),
        "intermediate": (170, 150, 70),
        "thick": (170, 70, 90),
    }[a]
    r = random.randint(45, 70)
    y.ellipse((128 - r, 128 - r, 128 + r, 128 + r), fill=z)
    r2 = random.randint(18, 35)
    y.ellipse((128 - r2, 128 - r2, 128 + r2, 128 + r2), fill=(30, 30, 30))
    for _ in range(6):
        p1 = (random.randint(20, 236), random.randint(20, 236))
        p2 = (random.randint(20, 236), random.randint(20, 236))
        c = tuple(max(0, min(255, v + random.randint(-25, 25))) for v in z)
        y.line((p1, p2), fill=c, width=random.randint(1, 3))
    n = np.random.randint(0, 18, (256, 256, 3), dtype=np.uint8)
    x = np.array(x)
    x = np.clip(x + n, 0, 255).astype(np.uint8)
    return x


def save_image(a, b):
    if CV2_OK:
        x = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        cv2.imwrite(b, x)
    else:
        Image.fromarray(a).save(b)


def ensure_dummy_dataset():
    make_dirs()
    a = 0
    for b in ["train", "test"]:
        for c in CLASSES:
            d = os.path.join("dataset", b, c)
            if os.path.isdir(d):
                e = [
                    f
                    for f in os.listdir(d)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                a += len(e)
    if a > 0:
        fill_missing_images()
        return

    for b, n in [("train", 24), ("test", 8)]:
        for c in CLASSES:
            d = os.path.join("dataset", b, c)
            for i in range(n):
                x = draw_dummy_image(c)
                y = os.path.join(d, f"{c}_{i}.png")
                save_image(x, y)


def fill_missing_images():
    for a, n in [("train", 12), ("test", 4)]:
        for b in CLASSES:
            c = os.path.join("dataset", a, b)
            os.makedirs(c, exist_ok=True)
            d = [
                f
                for f in os.listdir(c)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if len(d) == 0:
                for i in range(n):
                    x = draw_dummy_image(b)
                    y = os.path.join(c, f"{b}_{i}.png")
                    save_image(x, y)


def read_image(a, b=IMG_SIZE):
    if CV2_OK:
        x = cv2.imread(a)
        if x is None:
            raise ValueError(f"Image not found: {a}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (b, b))
    else:
        x = Image.open(a).convert("RGB").resize((b, b))
        x = np.array(x)
    return x


def preprocess_image(a, b=IMG_SIZE, c=False):
    x = read_image(a, b=b).astype(np.float32) / 255.0
    m = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    s = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - m) / s
    x = np.transpose(x, (2, 0, 1))
    if c and TORCH_OK:
        return torch.tensor(x, dtype=torch.float32)
    return x.astype(np.float32)


def collect_samples(a="train", b="multiclass"):
    ensure_dummy_dataset()
    x = []
    y = []
    z = BINARY_CLASSES if b == "binary" else CLASSES
    for i, c in enumerate(CLASSES):
        d = os.path.join("dataset", a, c)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if b == "binary" and c == "intermediate":
                continue
            p = os.path.join(d, f)
            q = 0 if (b == "binary" and c == "thin") else 1 if b == "binary" else i
            x.append(p)
            y.append(q)
    return x, y, z


def load_numpy_split(a="train", b="multiclass", c=SMALL_SIZE):
    x, y, z = collect_samples(a, b)
    p = []
    for f in x:
        q = preprocess_image(f, b=c, c=False)
        p.append(q.reshape(-1))
    return np.stack(p).astype(np.float32), np.array(y, dtype=np.int64), z


class MelanomaDataset(Dataset):
    def __init__(self, a="train", b="multiclass"):
        self.files, self.labels, self.names = collect_samples(a, b)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, a):
        x = preprocess_image(self.files[a], b=IMG_SIZE, c=True)
        y = torch.tensor(self.labels[a], dtype=torch.long)
        return x, y


def get_dataloaders(a="multiclass", b=8):
    if not TORCH_OK:
        x = load_numpy_split("train", a)
        y = load_numpy_split("test", a)
        return x, y
    x = MelanomaDataset("train", a)
    y = MelanomaDataset("test", a)
    z = DataLoader(x, batch_size=b, shuffle=True)
    p = DataLoader(y, batch_size=b, shuffle=False)
    return z, p
