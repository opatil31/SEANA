"""Frozen six-class probes; validation selection never examines test data."""

import copy

import torch
from torch import nn

from .metrics import evaluate_predictions


@torch.inference_mode()
def evaluate(backbone, head, loader, records, device):
    backbone.eval()
    head.eval()
    labels, predictions = [], []
    for image, label in loader:
        prediction = head(backbone(image.to(device)).flatten(1)).argmax(1)
        labels.extend(label.tolist())
        predictions.extend(prediction.cpu().tolist())
    metrics = evaluate_predictions(labels, predictions, records)
    rows = [
        dict(image_id=r["image_id"], patient_id=r["patient_id"], machine=r["machine"], label=y, prediction=p)
        for r, y, p in zip(records, labels, predictions)
    ]
    return metrics, rows


def train_probe(backbone, feature_dim, train_loader, validation_loader, validation_records, cfg, device):
    backbone.eval().requires_grad_(False)
    head = nn.Linear(feature_dim, 6).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay)
    if cfg.probe_selection == "validation_macro_f1" and validation_loader is None:
        raise ValueError("Validation probe selection needs held-out validation patients")
    history, best, selected_state, selected_epoch = [], -1.0, None, cfg.probe_epochs
    for epoch in range(cfg.probe_epochs):
        head.train()
        total, count = 0.0, 0
        for image, label in train_loader:
            with torch.no_grad():
                features = backbone(image.to(device)).flatten(1)
            loss = nn.functional.cross_entropy(head(features), label.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite linear probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(label)
            count += len(label)
        if not count:
            raise ValueError("No probe training images")
        row = dict(epoch=epoch + 1, train_loss=total / count)
        if cfg.probe_selection == "validation_macro_f1":
            metrics, _ = evaluate(backbone, head, validation_loader, validation_records, device)
            row["validation"] = metrics
            score = metrics["overall"]["macro_f1"]
            if score > best:
                best, selected_epoch = score, epoch + 1
                selected_state = copy.deepcopy(head.state_dict())
        history.append(row)
        print(f"[probe] epoch={epoch + 1} loss={total / count:.6f}")
    if selected_state is not None:
        head.load_state_dict(selected_state)
    return head, history, selected_epoch
