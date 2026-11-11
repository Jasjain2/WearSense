"""Training loop with early stopping on validation loss."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.models import ActivityClassifier

DEFAULT_CFG = {
    "lr": config.LEARNING_RATE,
    "weight_decay": config.WEIGHT_DECAY,
    "batch_size": config.BATCH_SIZE,
    "max_epochs": config.MAX_EPOCHS,
    "patience": config.EARLY_STOPPING_PATIENCE,
    "hidden_dim": config.HIDDEN_DIM,
    "dropout": config.DROPOUT,
    "seed": config.RANDOM_SEED,
}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray) -> torch.Tensor:
    """Logits for X with the model in eval mode, without gradients."""
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32))


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Softmax class probabilities [n, num_classes]."""
    return torch.softmax(predict_logits(model, X), dim=1).numpy()


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                input_dim: int, num_classes: int,
                cfg: dict | None = None) -> tuple[ActivityClassifier, dict]:
    """Train an ActivityClassifier with Adam + cross-entropy and early stopping
    on validation loss. The returned model holds the best-val-loss weights.

    cfg overrides keys of DEFAULT_CFG (lr, weight_decay, batch_size,
    max_epochs, patience, hidden_dim, dropout, seed).

    History: per-epoch lists "train_loss", "train_acc", "val_loss", "val_acc",
    plus "best_epoch" (1-based), "best_val_loss" and "epochs_run".
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    set_seed(cfg["seed"])

    model = ActivityClassifier(input_dim, num_classes, cfg["hidden_dim"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                       torch.as_tensor(y_train, dtype=torch.long))
    # BatchNorm can't train on a batch of 1; drop such a trailing batch.
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=len(ds) % cfg["batch_size"] == 1,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        history["train_loss"].append(total_loss / n)
        history["train_acc"].append(correct / n)

        val_logits = predict_logits(model, X_val)
        val_loss = loss_fn(val_logits, y_val_t).item()
        history["val_loss"].append(val_loss)
        history["val_acc"].append((val_logits.argmax(1) == y_val_t).float().mean().item())

        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_loss
    history["epochs_run"] = len(history["train_loss"])
    return model, history
