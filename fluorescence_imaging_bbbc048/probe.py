"""Seven-class frozen probes, with accuracy, balanced recall, macro-F1 and confusion."""

import torch
from torch import nn


def confusion_metrics(confusion):
    matrix = torch.as_tensor(confusion, dtype=torch.long)
    if matrix.shape != (7, 7) or (matrix < 0).any() or matrix.sum() == 0:
        raise ValueError("Expected a nonempty 7x7 confusion matrix")
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = matrix.diag().double()
    recall, precision = tp / support.clamp_min(1), tp / predicted.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    present = support > 0
    return dict(
        accuracy=float(tp.sum() / matrix.sum()),
        balanced_accuracy=float(recall[present].mean()),
        macro_f1=float(f1[present].mean()),
        per_class_recall=recall.tolist(),
        per_class_precision=precision.tolist(),
        per_class_support=support.tolist(),
        n_classes_present=int(present.sum()),
        confusion=matrix.tolist(),
    )


@torch.inference_mode()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    confusion = torch.zeros((7, 7), dtype=torch.long)
    for image, label in loader:
        prediction = head(backbone(image.to(device)).flatten(1)).argmax(1).cpu()
        confusion += torch.bincount(7 * label.cpu() + prediction, minlength=49).reshape(7, 7)
    return confusion_metrics(confusion)


def train_probe(backbone, feature_dim, train_loader, test_loader, cfg, device):
    backbone.eval().requires_grad_(False)
    head = nn.Linear(feature_dim, 7).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay)
    history, best_accuracy, best_balanced = [], None, None
    for epoch in range(cfg.probe_epochs):
        head.train()
        total, count = 0.0, 0
        for image, label in train_loader:
            with torch.no_grad():
                features = backbone(image.to(device)).flatten(1)
            loss = nn.functional.cross_entropy(head(features), label.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite linear-probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(label)
            count += len(label)
        if count == 0:
            raise ValueError("No probe training samples")
        row = dict(epoch=epoch + 1, train_loss=total / count)
        if cfg.probe_report_best:
            metrics = evaluate(backbone, head, test_loader, device)
            row["test"] = metrics
            best_accuracy = max(best_accuracy or 0, metrics["accuracy"])
            best_balanced = max(best_balanced or 0, metrics["balanced_accuracy"])
        history.append(row)
        print(f"[probe] epoch={epoch + 1} loss={total / count:.6f}")
    metrics = {
        "final": evaluate(backbone, head, test_loader, device),
        "best_accuracy_over_test_epochs": best_accuracy,
        "best_balanced_accuracy_over_test_epochs": best_balanced,
    }
    return head, metrics, history
