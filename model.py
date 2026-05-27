import pickle

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
except Exception:
    torch = None
    nn = None
    TORCH_OK = False


if TORCH_OK:
    class SimpleCNN(nn.Module):
        def __init__(self, a=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 28 * 28, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, a),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
else:
    class SimpleCNN:
        def __init__(self, a=2, b=3 * 32 * 32, c=64):
            rng = np.random.default_rng(42)
            self.num_classes = a
            self.input_dim = b
            self.hidden_dim = c
            self.w1 = rng.normal(0, 0.02, (b, c)).astype(np.float32)
            self.b1 = np.zeros((1, c), dtype=np.float32)
            self.w2 = rng.normal(0, 0.02, (c, a)).astype(np.float32)
            self.b2 = np.zeros((1, a), dtype=np.float32)

        def forward(self, x):
            h = x @ self.w1 + self.b1
            h = np.maximum(h, 0)
            y = h @ self.w2 + self.b2
            return y, h

        def predict_proba(self, x):
            y, _ = self.forward(x)
            y = y - np.max(y, axis=1, keepdims=True)
            z = np.exp(y)
            return z / np.sum(z, axis=1, keepdims=True)

        def fit(self, x, y, epochs=8, lr=0.01, batch_size=8):
            n = x.shape[0]
            for epoch in range(epochs):
                idx = np.random.permutation(n)
                x = x[idx]
                y = y[idx]
                for i in range(0, n, batch_size):
                    a = x[i : i + batch_size]
                    b = y[i : i + batch_size]
                    y1, h = self.forward(a)
                    y1 = y1 - np.max(y1, axis=1, keepdims=True)
                    p = np.exp(y1)
                    p = p / np.sum(p, axis=1, keepdims=True)
                    t = np.zeros_like(p)
                    t[np.arange(len(b)), b] = 1.0
                    g = (p - t) / len(b)
                    dw2 = h.T @ g
                    db2 = np.sum(g, axis=0, keepdims=True)
                    dh = g @ self.w2.T
                    dh[h <= 0] = 0
                    dw1 = a.T @ dh
                    db1 = np.sum(dh, axis=0, keepdims=True)
                    self.w2 -= lr * dw2
                    self.b2 -= lr * db2
                    self.w1 -= lr * dw1
                    self.b1 -= lr * db1

        def predict(self, x):
            p = self.predict_proba(x)
            return np.argmax(p, axis=1)


def get_binary_model():
    return SimpleCNN(2)


def get_multiclass_model():
    return SimpleCNN(3)


def get_device():
    return torch.device("cpu") if TORCH_OK else "cpu"


def save_checkpoint(a, b, c, d):
    x = {"class_names": c, "model_type": d, "torch": TORCH_OK}
    if TORCH_OK:
        x["model_state_dict"] = b.state_dict()
        torch.save(x, a)
    else:
        x["weights"] = {
            "w1": b.w1,
            "b1": b.b1,
            "w2": b.w2,
            "b2": b.b2,
            "num_classes": b.num_classes,
        }
        with open(a, "wb") as f:
            pickle.dump(x, f)


def load_checkpoint(a, b):
    if TORCH_OK:
        x = torch.load(a, map_location=get_device())
    else:
        with open(a, "rb") as f:
            x = pickle.load(f)
    y = get_binary_model() if b == "binary" else get_multiclass_model()
    if x.get("torch") and TORCH_OK:
        y.load_state_dict(x["model_state_dict"])
        y.eval()
    elif "weights" in x:
        y.w1 = x["weights"]["w1"]
        y.b1 = x["weights"]["b1"]
        y.w2 = x["weights"]["w2"]
        y.b2 = x["weights"]["b2"]
    return y, x["class_names"]