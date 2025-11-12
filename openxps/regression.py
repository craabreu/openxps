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

from .dynamical_variable import DynamicalVariable


class RBFPotential(nn.Module):
    r"""Radial basis function potential.

    .. math::
        U({\bf s}) = \sum_{m=1}^n w_m * \exp\left(
            -\frac{1}{2 \sigma_m^2} * r^2({\bf s}, {\bf c}_m)
        \right)

    where :math:`{\bf s}` is the extended phase-space position, :math:`{\bf c}_m` are
    the centers of the radial basis functions, :math:`\sigma_m` are the standard
    deviations, and :math:`w_m` are the weights.

    The gradient of the potential is given by:

    .. math::
        \nabla_{\bf s} U = -\frac{1}{2} \sum_{m=1}^n \frac{w_m}{\sigma_m^2} \exp\left(
            -\frac{1}{2 \sigma_m^2} * r^2({\bf s}, {\bf c}_m)
        \right) \nabla_{\bf s} r^2({\bf s}, {\bf c}_m)

    Parameters
    ----------
    in_dim
        The dimension of the extended phase-space position.
    M
        The number of radial basis functions.
    init_sigma
        The initial standard deviation of the radial basis functions.
    learn_centers
        Whether to learn the centers of the radial basis functions.
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


class ForceMatchingRegressor:
    r"""Potential regressor from sampled positions and forces.

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
    dynamical_variables: t.Sequence[DynamicalVariable]
        The dynamical variables to be used in the potential.
    num_kernels: int
        The number of kernels to be used in the potential.

    Keyword Arguments
    -----------------
    initial_bandwidth: float
        The initial bandwidth of the kernels.
    validation_fraction: float
        The fraction of the data to be used for validation.
    batch_size: int
        The batch size to be used for training.
    num_epochs: int
        The number of epochs to be used for training.
    learning_rate: float
        The learning rate to be used for training.
    weight_decay: float
        The weight decay to be used for training.
    seed: int
        The seed to be used for training.
    accelerator: str
        The accelerator to be used for training. Valid options can be found
        `here <https://lightning.ai/docs/pytorch/LTS/common/trainer.html#accelerator>`_.
    num_workers: int
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
            in_dim=n,
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

    def predict(self, positions: npt.ArrayLike) -> npt.ArrayLike:
        """Predict the potential at the given positions.

        Parameters
        ----------
        positions: npt.ArrayLike
            An array of shape (M, d) containing the positions at which to predict the
            potential.

        Returns
        -------
        potential: np.ndarray
            An array of shape (M,) containing the potential at the given positions.
        """
        X = torch.as_tensor(positions, dtype=torch.float32)
        with torch.no_grad():
            return self._model.f(X).detach().cpu().numpy()

    def get_parameters(self) -> dict[str, np.ndarray]:
        """Get the parameters of the potential regressor.

        Returns
        -------
        centers: np.ndarray
            The centers of the radial basis functions.
        sigmas: np.ndarray
            The bandwidths of the radial basis functions.
        weights: np.ndarray
            The weights of the radial basis functions.
        """
        return (
            self._model.f.c.detach().cpu().numpy(),
            self._model.f.logsig.exp().detach().cpu().numpy(),
            self._model.f.w.detach().cpu().numpy(),
        )
