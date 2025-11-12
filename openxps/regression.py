"""
.. module:: openxps.rbf
   :platform: Linux, MacOS, Windows
   :synopsis: Radial basis function regressor for free energy surface fitting.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import torch
from lightning import pytorch as pl
from numpy import typing as npt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bounds import NoBounds
from .dynamical_variable import DynamicalVariable


class RBFPotential(nn.Module):
    """Radial basis function potential.

    Parameters
    ----------
    dynamical_variables
        A sequence of dynamical variables defining the dimensions and bounds
        for the potential. The length of this sequence determines the input
        dimension.
    M
        The number of radial basis functions.
    init_sigma
        The initial standard deviation of the radial basis functions
        (same for all dimensions).
    learn_centers
        Whether to learn the centers of the radial basis functions.
    """

    def __init__(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        M: int = 256,
        init_sigma: float = 0.5,
        learn_centers: bool = True,
    ) -> None:
        in_dim = len(dynamical_variables)
        bounds = [
            (-1, 1)
            if isinstance(dv.bounds, NoBounds)
            else (dv.bounds.lower, dv.bounds.upper)
            for dv in dynamical_variables
        ]
        super().__init__()
        self.c = nn.Parameter(
            torch.stack([torch.randn(M) * (ub - lb) + lb for lb, ub in bounds], dim=-1),
            requires_grad=learn_centers,
        )
        self.logsig = nn.Parameter(torch.full((M, in_dim), float(np.log(init_sigma))))
        self.w = nn.Parameter(torch.zeros(M))
        self._length = nn.Parameter(
            torch.tensor([ub - lb for lb, ub in bounds], dtype=torch.float32),
            requires_grad=False,
        )

    def _delta2_fn(
        self, disp: torch.Tensor
    ) -> (
        torch.Tensor
    ):  # disp: (B,M,d) -> (B,M,d),   δ² = (L/π) sin²(π d/L) per dimension
        length = self._length[None, None, :]  # (1, 1, d) for broadcasting
        return ((length / np.pi) * torch.sin(np.pi * disp / length)) ** 2

    def _delta2_grad(
        self, disp: torch.Tensor
    ) -> torch.Tensor:  # ∇_x δ² = (L/π) sin(2πd/L) per dimension
        length = self._length[None, None, :]  # (1, 1, d) for broadcasting
        return (length / np.pi) * torch.sin(2 * np.pi * disp / length)

    def _phi(self, disp: torch.Tensor) -> torch.Tensor:  # (B,M,d) -> (B,M)
        delta2 = self._delta2_fn(disp)  # (B,M,d)
        sigma2 = torch.exp(2 * self.logsig)  # (M,d)
        # Sum over dimensions: sum_k (δ²_k / σ²_{m,k})
        return torch.exp(-0.5 * (delta2 / sigma2[None, :, :]).sum(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,d) -> (B,)
        disp = x[:, None, :] - self.c[None, :, :]
        return self._phi(disp) @ self.w

    def grad(self, x: torch.Tensor) -> torch.Tensor:  # (B,d)
        disp = x[:, None, :] - self.c[None, :, :]
        Phi = self._phi(disp)  # (B,M)
        delta2_grad = self._delta2_grad(disp)  # (B,M,d)
        sigma2 = torch.exp(2 * self.logsig)  # (M,d)
        # For each kernel m: w_m * φ_m * sum_k (1/σ²_{m,k} * ∇δ²_k)
        fac = (self.w[:, None] / sigma2)[None, :, :]  # (1,M,d)
        return -0.5 * (fac * Phi[:, :, None] * delta2_grad).sum(1)


class GradMatch(pl.LightningModule):
    def __init__(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        M: int = 256,
        lr: float = 2e-3,
        wd: float = 1e-4,
        init_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.f = RBFPotential(dynamical_variables, M, init_sigma)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )

    def _loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, F = batch
        return ((self.f.grad(x) + F) ** 2).mean()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss


class ForceMatchingRegressor:
    r"""Potential regressor from sampled position/force pairs.

    A potential function in a :math:`d`-dimensional variable space is approximated as:

    .. math::
        U({\bf s}) = \sum_{m=1}^n w_m \exp\left(
            -\frac{1}{2} \sum_{k=1}^d \frac{\delta_k^2(s_k - c_{m,k})}{\sigma_{m,k}^2}
        \right)

    where the :math:`n` weights :math:`w_m`, kernel bandwidths :math:`\sigma_m`, and
    kernel centers :math:`{\bf c}_m` are adjustable parameters. The displacement
    function :math:`\delta_k(x)` depends on the periodicity of the variable :math:`s_k`
    and is defined as:

    .. math::
        \delta_k(x) = \left\{\begin{array}{ll}
            x & \text{if }s_k\text{ is non-periodic} \\
            \frac{L_k}{\pi}\sin(\frac{\pi x}{L_k}) & \text{if }s_k\text{ has period }L_k
        \end{array}\right.

    Given :math:`N` position-force pairs :math:`({\bf s}_i, {\bf F}_i)`, the parameters
    are adjusted by minimizing the mean squared error loss:

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N \left\| {\bf f}({\bf s}_i) - {\bf F}_i \right\|_2,

    where :math:`{\bf f}({\bf s}) = -\nabla_{\bf s} U({\bf s})` is the predicted force.

    Parameters
    ----------
    dynamical_variables
        The dynamical variables to be used in the potential.
    num_kernels
        The number of kernels to be used in the potential.

    Keyword Arguments
    -----------------
    initial_bandwidth
        The initial bandwidth of the kernels.
    validation_fraction
        The fraction of the data to be used for validation.
    batch_size
        The batch size to be used for training.
    num_epochs
        The number of epochs to be used for training.
    learning_rate
        The learning rate to be used for training.
    weight_decay
        The weight decay to be used for training.
    seed
        The seed to be used for training.
    accelerator
        The accelerator to be used for training. Valid options can be found
        `here <https://lightning.ai/docs/pytorch/LTS/common/trainer.html#accelerator>`_.
    num_workers
        The number of data loading workers to be used for training.
    """

    def __init__(  # noqa: PLR0913
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        num_kernels: int,
        *,
        initial_bandwidth: float = 1.0,
        validation_fraction: float = 0.1,
        batch_size: int = 256,
        num_epochs: int = 50,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        accelerator: str = "auto",
        num_workers: int = 0,
    ):
        self._dynamical_variables = tuple(
            dv.in_md_units() for dv in dynamical_variables
        )
        self._init_sigma = initial_bandwidth
        self._val_frac = validation_fraction
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._lr = learning_rate
        self._wd = weight_decay
        self._seed = seed
        self._accelerator = accelerator
        self._num_workers = num_workers
        self._M = num_kernels

    def fit(self, positions: npt.ArrayLike, forces: npt.ArrayLike) -> None:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        X = torch.as_tensor(positions, dtype=torch.float32)
        F = torch.as_tensor(forces, dtype=torch.float32)
        N, n = X.shape
        idx = torch.randperm(N)
        nval = int(round(self._val_frac * N))
        val_idx, tr_idx = idx[:nval], idx[nval:]
        dl = DataLoader(
            TensorDataset(X[tr_idx], F[tr_idx]),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )
        if nval > 0:
            dlv = DataLoader(
                TensorDataset(X[val_idx], F[val_idx]),
                batch_size=self._batch_size,
                num_workers=self._num_workers,
            )
        else:
            dlv = None
        self._model = GradMatch(
            dynamical_variables=self._dynamical_variables,
            M=self._M,
            lr=self._lr,
            wd=self._wd,
            init_sigma=self._init_sigma,
        )
        trainer = pl.Trainer(
            max_epochs=self._num_epochs,
            accelerator=self._accelerator,
            devices="auto",
            logger=False,
            enable_progress_bar=True,
        )
        trainer.fit(self._model, dl, dlv)

    def predict(self, positions: npt.ArrayLike) -> np.ndarray:
        """Predict the potential at the given positions.

        Parameters
        ----------
        positions
            An array of shape (M, d) containing the positions at which to predict the
            potential.

        Returns
        -------
        potential
            An array of shape (M,) containing the potential at the given positions.
        """
        X = torch.as_tensor(positions, dtype=torch.float32)
        with torch.no_grad():
            return self._model.f(X).detach().cpu().numpy()

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the parameters of the potential regressor.

        Returns
        -------
        centers
            The centers of the radial basis functions, shape (M, d).
        sigmas
            The bandwidths of the radial basis functions per dimension,
            shape (M, d).
        weights
            The weights of the radial basis functions, shape (M,).
        """
        return (
            self._model.f.c.detach().cpu().numpy(),
            self._model.f.logsig.exp().detach().cpu().numpy(),
            self._model.f.w.detach().cpu().numpy(),
        )
