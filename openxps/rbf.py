import numpy as np
import torch
from lightning import pytorch as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import typing as t

from openxps.dynamical_variable import DynamicalVariable


# -------- model --------
class RBFPotential(nn.Module):
    """
    U(x) = Σ_m w_m * exp( -0.5 * r2(x,c_m) / σ_m^2 )
    Generic gradient (works for any r2): ∇U = -0.5 Σ_m (w_m/σ_m^2) φ_m * ∇_x r2
    """

    def __init__(self, in_dim, M=256, init_sigma=0.5, learn_centers=True):
        super().__init__()
        # default centers on [-π, π] for periodic kernels; change if you use Euclidean
        self.c = nn.Parameter(
            (2 * torch.rand(M, in_dim) - 1) * np.pi, requires_grad=learn_centers
        )
        self.logsig = nn.Parameter(torch.full((M,), float(np.log(init_sigma))))
        self.w = nn.Parameter(torch.zeros(M))

    @staticmethod
    def _r2_fn(d):  # d: (B,M,d) -> (B,M),   r2 = 4 Σ sin^2((d_i)/2)
        return 4 * (torch.sin(d / 2) ** 2).sum(-1)

    @staticmethod
    def _r2_grad(d):  # ∇_x r2 = 2 sin(d)
        return 2 * torch.sin(d)

    def _phi(self, d):  # (B,M,d) -> (B,M)
        sigma2 = torch.exp(2 * self.logsig)  # (M,)
        return torch.exp(-0.5 * self._r2_fn(d) / sigma2)

    def forward(self, x):  # (B,d) -> (B,)
        d = x[:, None, :] - self.c[None, :, :]
        return self._phi(d) @ self.w

    def grad(self, x):  # (B,d)
        d = x[:, None, :] - self.c[None, :, :]
        Phi = self._phi(d)  # (B,M)
        sigma2 = torch.exp(2 * self.logsig)  # (M,)
        fac = (self.w / sigma2)[None, :, None]  # (1,M,1)
        # generic: -0.5 * (w/σ^2) * φ * ∇r2
        return -0.5 * (fac * Phi[:, :, None] * self._r2_grad(d)).sum(1)


# -------- Lightning wrapper --------
class GradMatch(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        M=256,
        lr=2e-3,
        wd=1e-4,
        init_sigma=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.f = RBFPotential(in_dim, M, init_sigma)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )

    def _loss(self, batch):
        x, F = batch
        return ((self.f.grad(x) + F) ** 2).mean()

    def training_step(self, batch, _):
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss


# -------- trainer --------
def train_potential_from_forces(  # noqa: PLR0913
    X,
    F,
    *,
    val_frac=0.1,
    batch=256,
    epochs=200,
    lr=2e-3,
    wd=1e-4,
    seed=0,
    accelerator="auto",
    num_workers=0,
    M=256,
    init_sigma=1.0,
):
    """X: (N,n), F: (N,n) with F = -∇U. Random train/val split via val_frac."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    F = torch.as_tensor(F, dtype=torch.float32)
    N, n = X.shape
    idx = torch.randperm(N)
    nval = int(round(val_frac * N))
    val_idx, tr_idx = idx[:nval], idx[nval:]
    dl = DataLoader(
        TensorDataset(X[tr_idx], F[tr_idx]),
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
    )
    if nval > 0:
        dlv = DataLoader(
            TensorDataset(X[val_idx], F[val_idx]),
            batch_size=batch,
            num_workers=num_workers,
        )
    else:
        dlv = None
    model = GradMatch(
        in_dim=n,
        M=M,
        lr=lr,
        wd=wd,
        init_sigma=init_sigma,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices="auto",
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, dl, dlv)
    return model
