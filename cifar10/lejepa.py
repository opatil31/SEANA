"""SIGReg and LeJEPA loss adapted from the project microscopy runner.

Upstream: Randall Balestriero and Yann LeCun, galilai-group/lejepa.
CC BY-NC 4.0; see third_party/lejepa-LICENSE.txt and THIRD_PARTY_NOTICES.md.
This module is a single-process adaptation.
"""

import torch
from torch import nn


class EppsPulleyStat(nn.Module):
    """Epps-Pulley empirical-characteristic-function normality statistic.

    Verbatim port of lejepa/univariate/epps_pulley.py::EppsPulley (world_size == 1):
        T(x) = N * sum_i w_i * [ (mean_n cos(t_i x_n) - phi_i)^2 + (mean_n sin(t_i x_n))^2 ]
    with t_i = linspace(0, t_max, n_points), phi_i = exp(-t_i^2 / 2) the standard-normal
    characteristic function, and w_i the trapezoid weights (doubled for the folded negative
    half, halved at the endpoints) already multiplied by phi_i. Deviation from N(0,1) in the
    projected samples x_n makes T large; T == 0 iff the samples match a standard normal CF.
    """

    def __init__(self, n_points: int = 17, t_max: float = 3.0):
        super().__init__()
        assert int(n_points) % 2 == 1, "n_points must be odd"
        t = torch.linspace(0.0, float(t_max), int(n_points), dtype=torch.float32)
        dt = float(t_max) / (int(n_points) - 1)
        weights = torch.full((int(n_points),), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt  # half-weight at the folded endpoints
        phi = torch.exp(-0.5 * t * t)  # standard-normal CF at t_i
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)  # w_i folds in phi_i, as in the repo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, N, K) -- N samples (the empirical-CF batch dim, at -2), K slices (at -1).
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, P)
        cos_mean = torch.cos(x_t).mean(dim=-3)  # (*, K, P) -- mean over samples
        sin_mean = torch.sin(x_t).mean(dim=-3)  # (*, K, P)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * float(N)  # (*, K)


class SlicedNormalityTest(nn.Module):
    """SIGReg: average a univariate normality test over num_slices random unit directions.

    Verbatim port of lejepa/multivariate/slicing.py::SlicingUnivariateTest (reduction='mean',
    gaussian sampler, no clipping): draw A ~ N(0, I) of shape (D, K), L2-normalise its columns
    to unit directions, project x @ A -> (*, N, K), apply the univariate test, and average.
    """

    def __init__(self, univariate_test: nn.Module, num_slices: int = 1024):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = int(num_slices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, N, D)
        D = x.size(-1)
        A = torch.randn(D, self.num_slices, device=x.device, dtype=x.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)  # unit-norm columns (unit directions)
        stats = self.univariate_test(x @ A)  # (*, K)
        return stats.mean()


class LeJEPALoss(nn.Module):
    """Upstream weighted LeJEPA objective, with an explicit legacy weighting option."""

    def __init__(
        self,
        num_slices: int = 1024,
        num_points: int = 17,
        sigreg_weight: float = 0.05,
        objective: str = "upstream",
    ):
        super().__init__()
        self.sigreg = SlicedNormalityTest(
            EppsPulleyStat(n_points=int(num_points)), num_slices=int(num_slices)
        )
        self.sigreg_weight = float(sigreg_weight)
        if objective not in ("upstream", "legacy"):
            raise ValueError("objective must be upstream or legacy")
        self.objective = objective
        self.last_inv = float("nan")  # stashed for logging only
        self.last_sig = float("nan")

    def forward(self, zs: torch.Tensor, n_global: int = 2) -> torch.Tensor:
        # zs: (V, N, D) stacked view projections. Prediction center uses only the GLOBAL views
        # (Eq. 8); with n_global == V this reduces to MINIMAL.md's proj.mean(0).
        ng = max(1, min(int(n_global), zs.size(0)))
        center = zs[:ng].mean(dim=0, keepdim=True)  # (1, N, D)
        inv = (zs - center).square().mean()  # mean over V, N, D (MINIMAL.md)
        sig = self.sigreg(zs)  # (1/V) sum_v SIGReg(z_v)
        self.last_inv = float(inv.detach())
        self.last_sig = float(sig.detach())
        inv_weight = 1.0 - self.sigreg_weight if self.objective == "upstream" else 1.0
        return self.sigreg_weight * sig + inv_weight * inv
