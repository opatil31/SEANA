"""Frozen-backbone linear probing on noisy train and clean test images."""

import torch
from torch import nn


@torch.inference_mode()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    correct = total = 0
    for x, y in loader:
        features = backbone(x.to(device)).flatten(1)
        prediction = head(features).argmax(1)
        correct += int((prediction == y.to(device)).sum())
        total += len(y)
    if not total:
        raise ValueError("Cannot evaluate an empty test set")
    return correct / total


def train_probe(backbone, feature_dim, train_loader, test_loader, cfg, device, *, train_preprocess=None):
    """Fit the head, optionally preprocessing training batches only.

    The callback consumes and returns normalized image batches. Evaluation
    always sends clean test images directly to the frozen backbone.
    """
    backbone.requires_grad_(False)
    backbone.eval()  # freeze BatchNorm running statistics as well as parameters
    head = nn.Linear(feature_dim, cfg.num_classes).to(device)
    optimizer = torch.optim.SGD(
        head.parameters(), lr=cfg.probe_lr, momentum=0.9, weight_decay=cfg.probe_weight_decay
    )
    step_schedule = getattr(cfg, "probe_schedule_interval", "epoch") == "step"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.probe_epochs * (len(train_loader) if step_schedule else 1))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history = []
    best = None
    for epoch in range(cfg.probe_epochs):
        head.train()
        total_loss = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if train_preprocess is not None:
                    x = train_preprocess(x)
                features = backbone(x).flatten(1)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = nn.functional.cross_entropy(head(features), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite linear-probe loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step_schedule:
                scheduler.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)
        if not count:
            raise ValueError("Cannot fit a probe on an empty training set")
        row = {"epoch": epoch + 1, "train_loss": total_loss / count}
        # Optional diagnostic matching the old runner. Never used to select the
        # saved head; the default protocol only accesses the test set at the end.
        if cfg.probe_report_best:
            acc = evaluate(backbone, head, test_loader, device)
            row["clean_test_acc"] = acc
            best = acc if best is None else max(best, acc)
        history.append(row)
        if not step_schedule:
            scheduler.step()
        print(f"[probe] epoch={epoch + 1} loss={row['train_loss']:.6f}")
    final = evaluate(backbone, head, test_loader, device)
    return head, {"probe_final_clean_acc": final, "probe_best_clean_acc": best}, history
