"""Approach 2: MobileNetV2 transfer learning (PyTorch)."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_avocado_classifier(num_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_f, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
    return model


def unfreeze_last_blocks(model, n_blocks=3):
    for blk in model.features[-n_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True


def _run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def fit(model, train_loader, val_loader, class_weights=None,
        epochs=25, lr=1e-3, patience=5, finetune_after=None):
    """Cross-entropy + Adam, early stopping on validation loss."""
    model.to(DEVICE)
    w = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best, best_loss, bad = None, np.inf, 0
    history = []
    for ep in range(1, epochs + 1):
        if finetune_after and ep == finetune_after + 1:
            unfreeze_last_blocks(model)
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr / 10)
            print("-> fine-tuning last 3 backbone blocks")
        tl, ta = _run_epoch(model, train_loader, criterion, opt)
        vl, va = _run_epoch(model, val_loader, criterion)
        history.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"epoch {ep:02d} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f}")
        if vl < best_loss - 1e-4:
            best_loss, best, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break
    model.load_state_dict(best)
    return model, history


@torch.no_grad()
def predict_loader(model, loader):
    model.eval().to(DEVICE)
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(DEVICE)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return ys, ps


@torch.no_grad()
def latency_ms(model, device="cpu", runs=50):
    m = copy.deepcopy(model).eval().to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(5):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000
