"""Stage II - the lightweight teacher that turns hard labels into soft ones.

Three input features, two hidden layers of 64, one sigmoid output: 4,481
parameters. Trained with binary cross-entropy, Adam, learning rate 1e-3, batch
256, for 100 epochs, keeping the weights of the epoch with the best validation
accuracy. The continuous output of the retained model is what Stage III fits.

Every number here matches the reported experiment; the architecture and the
checkpoint rule are the ones described in Section II-B.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TeacherMLP(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dims=(64, 64), output_dim: int = 1):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _accuracy(model, X, y, device) -> float:
    model.eval()
    pred = model(torch.as_tensor(X, dtype=torch.float32, device=device))
    return ((pred > 0.5).float() ==
            torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)
            ).float().mean().item()


def train_teacher(X_train, y_train, X_val, y_val, cfg: dict, seed: int | None = None):
    """Fit the teacher and return (predict_fn, history)."""
    tcfg = cfg["teacher"]
    seed = cfg["sampling"]["seed"] if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TeacherMLP(tcfg["input_dim"], tuple(tcfg["hidden_dims"]),
                       tcfg["output_dim"]).to(device)
    opt = optim.Adam(model.parameters(), lr=tcfg["lr"])
    criterion = nn.BCELoss()

    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                      torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1)),
        batch_size=tcfg["batch_size"], shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_loss = running / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.as_tensor(X_val, dtype=torch.float32, device=device))
            val_loss = criterion(
                val_pred,
                torch.as_tensor(y_val, dtype=torch.float32,
                                device=device).view(-1, 1)).item()
        acc = _accuracy(model, X_val, y_val, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"teacher: {parameter_count(model)} parameters | best val acc {best_acc:.4f}")

    @torch.no_grad()
    def predict(X):
        model.eval()
        out = model(torch.as_tensor(X, dtype=torch.float32, device=device))
        return out.cpu().numpy().ravel()

    return predict, history, model
