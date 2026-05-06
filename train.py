import os

import numpy as np

from data import BINARY_CLASSES, CLASSES, get_dataloaders
from model import TORCH_OK, get_binary_model, get_device, get_multiclass_model, save_checkpoint

if TORCH_OK:
    import torch
    import torch.nn as nn
    import torch.optim as optim


def run_epoch(a, b, c, d, e):
    if e == "train":
        a.train()
    else:
        a.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.set_grad_enabled(e == "train"):
        for x, y in d:
            x = x.to(c)
            y = y.to(c)
            z = a(x)
            loss = b(z, y)
            if e == "train":
                optimizer = getattr(run_epoch, "_opt")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            p = torch.argmax(z, dim=1)
            total += y.size(0)
            correct += (p == y).sum().item()
            loss_sum += loss.item() * y.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def train_with_torch(a="binary", b=3, c=0.001):
    d = get_device()
    e, f = get_dataloaders(a)
    g = get_binary_model() if a == "binary" else get_multiclass_model()
    g = g.to(d)
    h = nn.CrossEntropyLoss()
    i = optim.Adam(g.parameters(), lr=c)
    run_epoch._opt = i

    print(f"\nTraining {a} model with PyTorch on {d}...")
    for j in range(b):
        train_loss, train_acc = run_epoch(g, h, d, e, "train")
        test_loss, test_acc = run_epoch(g, h, d, f, "eval")
        print(
            f"Epoch {j + 1}/{b} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"test loss: {test_loss:.4f} | test acc: {test_acc:.4f}"
        )

    k = "models/binary_model.pth" if a == "binary" else "models/multiclass_model.pth"
    l = BINARY_CLASSES if a == "binary" else CLASSES
    save_checkpoint(k, g, l, a)
    print(f"Saved model: {k}")
    print(f"{a} accuracy: {test_acc:.4f}")
    return k


def train_with_numpy(a="binary", b=8, c=0.01):
    (x1, y1, _), (x2, y2, z) = get_dataloaders(a)
    m = get_binary_model() if a == "binary" else get_multiclass_model()
    print(f"\nTraining {a} model with NumPy fallback on CPU...")
    m.fit(x1, y1, epochs=b, lr=c, batch_size=8)
    p1 = m.predict(x1)
    p2 = m.predict(x2)
    train_acc = float(np.mean(p1 == y1))
    test_acc = float(np.mean(p2 == y2))
    print(f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
    k = "models/binary_model.pth" if a == "binary" else "models/multiclass_model.pth"
    save_checkpoint(k, m, z, a)
    print(f"Saved model: {k}")
    print(f"{a} accuracy: {test_acc:.4f}")
    return k


def train_model(a="binary", b=3, c=0.001):
    os.makedirs("models", exist_ok=True)
    if TORCH_OK:
        return train_with_torch(a, b=b, c=c)
    return train_with_numpy(a, b=max(b * 3, 8), c=0.01)


def train_all():
    a = []
    if not os.path.exists("models/binary_model.pth"):
        a.append(train_model("binary"))
    if not os.path.exists("models/multiclass_model.pth"):
        a.append(train_model("multiclass"))
    return a


if __name__ == "__main__":
    train_model("binary")
    train_model("multiclass")
