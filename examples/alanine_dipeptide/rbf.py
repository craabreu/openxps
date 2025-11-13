# rbf_grad_periodic_min.py
import numpy as np
import torch
from lightning import pytorch as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class RBFPotentialPeriodic(nn.Module):
    # f(x)=Σ w_m exp(-0.5 r2/σ_m^2)+a^T x+b, r2=4 Σ sin^2((x-c)/2)  (2π-periodic)
    def __init__(self, in_dim, M=256, init_sigma=1.0, learn_centers=True):
        super().__init__()
        self.c = nn.Parameter(torch.randn(M, in_dim) * 0.5, requires_grad=learn_centers)
        self.logsig = nn.Parameter(torch.full((M,), float(np.log(init_sigma))))
        self.w = nn.Parameter(torch.zeros(M))
        self.a = nn.Parameter(torch.zeros(in_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def _phi(self, x):  # x: (B,d) -> Phi: (B,M)
        d = x[:, None, :] - self.c[None, :, :]  # (B,M,d)
        r2 = 4 * (torch.sin(d / 2) ** 2).sum(-1)  # (B,M)
        sigma2 = torch.exp(2 * self.logsig)  # (M,)
        return torch.exp(-0.5 * r2 / sigma2)  # (B,M)

    def forward(self, x):  # (B,d) -> (B,)
        Phi = self._phi(x)
        return Phi @ self.w + x @ self.a + self.b

    def grad(self, x):  # analytic ∇f(x): (B,d)
        d = x[:, None, :] - self.c[None, :, :]  # (B,M,d)
        Phi = self._phi(x)  # (B,M)
        sigma2 = torch.exp(2 * self.logsig)  # (M,)
        fac = (self.w / sigma2)[None, :, None]  # (1,M,1)
        return -(fac * Phi[:, :, None] * torch.sin(d)).sum(1) + self.a


class GradMatch(pl.LightningModule):
    def __init__(self, in_dim, M=256, lr=2e-3, wd=1e-4, init_sigma=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.f = RBFPotentialPeriodic(in_dim, M, init_sigma)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )

    def _step(self, batch, split):
        x, f = batch
        gp = self.f.grad(x)
        loss = ((gp + f) ** 2).mean()
        self.log(f"{split}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")


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
    M=256,
    init_sigma=1.0,
    num_workers=0,
):
    """
    X: (N,n) inputs, F: (N,n) force targets with F = -∇U.
    val_frac: fraction of samples held out for validation (0 <= val_frac < 1).
    num_workers: dataloader workers used for both train/val loaders.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    F = torch.as_tensor(F, dtype=torch.float32)
    N = len(X)
    n = X.shape[1]
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0,1).")

    idx = torch.randperm(N)
    nval = int(round(val_frac * N))
    if nval > 0:
        val_idx = idx[:nval]
        tr_idx = idx[nval:]
        dlv = DataLoader(
            TensorDataset(X[val_idx], F[val_idx]),
            batch_size=batch,
            num_workers=num_workers,
        )
    else:
        val_idx = torch.tensor([], dtype=torch.long)
        tr_idx = idx
        dlv = None

    dl = DataLoader(
        TensorDataset(X[tr_idx], F[tr_idx]),
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
    )

    model = GradMatch(in_dim=n, M=M, lr=lr, wd=wd, init_sigma=init_sigma)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices="auto",
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, dl, dlv)
    return model


# ---- tiny demo ----
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    n, N = 3, 2000
    C = torch.randn(6, n) * 0.5
    w = torch.tensor([1.2, -0.8, 0.6, -1.0, 0.9, -0.4])
    sig = 0.7
    a = torch.tensor([0.2, -0.1, 0.05])

    def f_true(x):
        d = x[:, None, :] - C[None, :, :]
        r2 = 4 * (torch.sin(d / 2) ** 2).sum(-1)
        Phi = torch.exp(-0.5 * r2 / (sig**2))
        return Phi @ w + x @ a

    X = torch.randn(N, n).requires_grad_(True)
    G = torch.autograd.grad(f_true(X), X, torch.ones(N))[0]
    F = (-G).detach().numpy()
    X = X.detach().numpy()
    model = train_potential_from_forces(X, F, val_frac=0.15, M=256, epochs=100)
    with torch.no_grad():
        Xt = torch.randn(4, n)
        print(model.f(Xt))
        print(model.f.grad(Xt))
