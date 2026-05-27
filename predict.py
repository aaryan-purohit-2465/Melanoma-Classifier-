import os

import numpy as np

from data import IMG_SIZE, SMALL_SIZE, preprocess_image
from model import TORCH_OK, get_device, load_checkpoint

if TORCH_OK:
    import torch


def load_model(a="binary"):
    b = "models/binary_model.pth" if a == "binary" else "models/multiclass_model.pth"
    if not os.path.exists(b):
        raise FileNotFoundError(f"Model not found: {b}")
    c, d = load_checkpoint(b, a)
    e = get_device()
    if TORCH_OK:
        c.to(e)
        c.eval()
    return c, d, e


def predict_image(a, b="binary"):
    c, d, e = load_model(b)
    if TORCH_OK:
        x = preprocess_image(a, b=IMG_SIZE, c=True).unsqueeze(0).to(e)
        with torch.no_grad():
            y = c(x)
            z = torch.softmax(y, dim=1)[0]
            p = int(torch.argmax(z).item())
            q = z.cpu().tolist()
    else:
        x = preprocess_image(a, b=SMALL_SIZE, c=False).reshape(1, -1).astype(np.float32)
        z = c.predict_proba(x)[0]
        p = int(np.argmax(z))
        q = z.tolist()
    r = d[p]
    s = float(q[p])
    return {"predicted_class": r, "confidence": s, "scores": q}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python predict.py <image_path> <binary|multiclass>")
        raise SystemExit(1)
    a = predict_image(sys.argv[1], sys.argv[2])
    print(a)