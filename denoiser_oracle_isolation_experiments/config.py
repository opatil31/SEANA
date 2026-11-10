"""Rebuttal recipes with explicit estimator and resampling controls."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from cifar10.config import Config, load_config as load_base_config
from denoisers import N2N_IMPLEMENTATION

EXPERIMENT = "denoiser_oracle_isolation"
TABLE_METHODS = ("byol", "simclr", "vicreg")
TABLE_ARMS = ("n2n", "n2n_resample", "nsp", "nsp_resample", "seana", "oracle_postcrop", "baseline")
ARM_TREATMENTS = {
    "baseline": "baseline",
    "n2n": "n2n",
    "n2n_resample": "n2n",
    "nsp": "nsp",
    "nsp_resample": "nsp",
    "seana": "seana",
    "oracle_postcrop": "baseline",
    "oracle_precrop": "baseline",
    "drunet_vst_resample": "seana",
}


@dataclass
class IsolationConfig(Config):
    arm: str = "baseline"
    probe_report_best: bool = True
    treatment_checkpoint: str = ""

    def validate(self):
        super().validate()
        if self.arm not in ARM_TREATMENTS:
            raise ValueError(f"arm must be one of {tuple(ARM_TREATMENTS)}")
        if self.treatment != ARM_TREATMENTS[self.arm]:
            raise ValueError("arm and base treatment disagree")
        if self.treatment_checkpoint and self.treatment == "baseline":
            raise ValueError("Baseline/oracle arms do not use fitted treatment checkpoints")
        return self

    def resolved(self):
        data = super().resolved()
        # Record the comparison's N2N method on every arm so compatible arms
        # share a table, while old and new N2N comparisons remain separate.
        data["n2n_implementation"] = N2N_IMPLEMENTATION
        data["experiment"] = EXPERIMENT
        data["oracle_order"] = {
            "oracle_precrop": "before_augmentation",
            "oracle_postcrop": "after_augmentation",
        }.get(self.arm)
        data["resampling"] = (
            "known_noise"
            if self.arm.endswith("resample") or self.arm.startswith("oracle_")
            else "learned_vst_gaussian"
            if self.arm == "seana"
            else "none"
        )
        return data

    def run_name(self):
        return self.arm + "_" + super().run_name()


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({key: value for key, value in overrides.items() if value is not None})
    arm = values.pop("arm", "baseline")
    checkpoint = values.pop("treatment_checkpoint", "")
    if arm not in ARM_TREATMENTS:
        raise ValueError(f"Unknown arm: {arm}")
    expected = ARM_TREATMENTS[arm]
    if "treatment" in values and values["treatment"] != expected:
        raise ValueError("Do not override the base treatment independently of the isolation arm")
    values["treatment"] = expected
    values.setdefault("probe_report_best", True)
    base = load_base_config(**values)
    return IsolationConfig(**asdict(base), arm=arm, treatment_checkpoint=checkpoint).validate()


def from_resolved(values):
    known = {field.name for field in fields(IsolationConfig)}
    return load_config(**{key: value for key, value in values.items() if key in known})
