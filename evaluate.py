import os

import matplotlib.pyplot as plt
import numpy as np

from data import get_dataloaders
from model import TORCH_OK, get_device, load_checkpoint

if TORCH_OK:
    import torch


def confusion_matrix(a, b):
    n = len(b)
    x = np.zeros((n, n), dtype=int)
    for y, z in a:
        x[y, z] += 1
    return x


def save_plot(a, b, c):
    plt.figure(figsize=(6, 5))
    plt.imshow(a, cmap="Blues")
    plt.title(c)
    plt.xticks(range(len(b)), b)
    plt.yticks(range(len(b)), b)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(b)):
        for j in range(len(b)):
            plt.text(j, i, str(a[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    x = os.path.join("results", f"{c.lower().replace(' ', '_')}.png")
    plt.savefig(x)
    plt.close()
    return x


def evaluate_model(a="binary"):
    b = "models/binary_model.pth" if a == "binary" else "models/multiclass_model.pth"
    if not os.path.exists(b):
        raise FileNotFoundError(f"Model not found: {b}")

    c, d = load_checkpoint(b, a)
    g = []

    if TORCH_OK:
        e = get_device()
        c.to(e)
        c.eval()
        _, f = get_dataloaders(a)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in f:
                x = x.to(e)
                y = y.to(e)
                z = c(x)
                p = torch.argmax(z, dim=1)
                for i in range(len(y)):
                    g.append((int(y[i].item()), int(p[i].item())))
                correct += (p == y).sum().item()
                total += y.size(0)
        h = correct / max(total, 1)
    else:
        (_, _, _), (x, y, _) = get_dataloaders(a)
        p = c.predict(x)
        for i in range(len(y)):
            g.append((int(y[i]), int(p[i])))
        h = float(np.mean(p == y))

    j = confusion_matrix(g, d)
    k = save_plot(j, d, f"{a} confusion matrix")
    print(f"{a} accuracy: {h:.4f}")
    print("Confusion matrix:")
    print(j)
    print(f"Saved plot: {k}")
    return h, j, k


if __name__ == "__main__":
    evaluate_model("binary")
    evaluate_model("multiclass")
