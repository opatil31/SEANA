"""Strict frozen-backbone loading for release checkpoints and explicit legacy metadata."""

import hashlib
import io
from pathlib import Path

from cifar10.config import BACKBONES, CONDITIONS, METHODS


def load_backbone(path, device, legacy_metadata=None):
    import torch
    from cifar10.models import build_backbone

    path = Path(path)
    data = path.read_bytes()
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    legacy_metadata = legacy_metadata or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected a checkpoint dictionary")
    kind = payload.get("kind")
    if kind is not None:
        if kind not in ("backbone", "ssl") or payload.get("format_version") != 1:
            raise ValueError("Expected a release-format backbone or SSL checkpoint")
        config = payload["config"]
        if config.get("dataset", "cifar10") != "cifar10":
            raise ValueError("This analysis is for CIFAR-10 checkpoints")
        info = {
            "backbone": config["backbone"],
            "condition": config["condition"],
            "method": config["method"],
            "arm": config.get("arm", config["treatment"]),
            "training_seed": config["seed"],
        }
        for key, value in legacy_metadata.items():
            if info.get(key) != value:
                raise ValueError(f"Explicit metadata disagrees with checkpoint: {key}")
        state = payload["state_dict"]
        training_config = config
        smoke = bool(config.get("smoke", False))
        format_name = f"release_{kind}"
    else:
        required = {"backbone", "condition", "method", "arm", "training_seed"}
        if set(legacy_metadata) != required:
            raise ValueError(
                "Legacy weights need explicit backbone, condition, method, arm and training_seed"
            )
        info = dict(legacy_metadata)
        state, training_config, smoke = payload, None, False
        format_name = "legacy_state_dict"
    if (
        info["backbone"] not in BACKBONES
        or info["condition"] not in CONDITIONS
        or info["method"] not in METHODS
    ):
        raise ValueError("Unsupported backbone, CIFAR condition or SSL method")
    if not isinstance(info["arm"], str) or not info["arm"].strip():
        raise ValueError("A nonempty training arm is required")
    if (
        not isinstance(info["training_seed"], int)
        or isinstance(info["training_seed"], bool)
        or info["training_seed"] < 0
    ):
        raise ValueError("training_seed must be a nonnegative integer")
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing model state")
    # Only the online encoder: backbone_momentum.* never matches backbone.*.
    if any(key.startswith("backbone.") for key in state):
        state = {
            key[len("backbone.") :]: value for key, value in state.items() if key.startswith("backbone.")
        }
    elif kind == "ssl":
        raise ValueError("SSL checkpoint is missing online backbone.* weights")
    model, feature_dim = build_backbone(info["backbone"])
    if kind == "backbone" and payload.get("feature_dim") != feature_dim:
        raise ValueError("Backbone feature dimension differs from checkpoint metadata")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, {
        **info,
        "checkpoint_name": path.name,
        "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
        "checkpoint_format": format_name,
        "feature_dim": feature_dim,
        "training_config": training_config,
        "checkpoint_smoke": smoke,
    }
