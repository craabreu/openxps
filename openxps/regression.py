"""
.. module:: openxps.rbf
   :platform: Linux, MacOS, Windows
   :synopsis: Radial basis function regressor for free energy surface fitting.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import tempfile
import typing as t

import numpy as np
import torch
from lightning import pytorch as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bounds import PeriodicBounds
from .dynamical_variable import DynamicalVariable

if t.TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:

    class ArrayLike(t.Protocol): ...


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
        super().__init__()
        in_dim = len(dynamical_variables)
        length_over_pi, periodic, random_points = [], [], []
        for dv in dynamical_variables:
            length_over_pi.append(dv.bounds.length / np.pi)
            periodic.append(isinstance(dv.bounds, PeriodicBounds))
            random_points.append(torch.randn(M) * dv.bounds.length + dv.bounds.lower)

        self.c = nn.Parameter(
            torch.stack(random_points, dim=-1), requires_grad=learn_centers
        )
        self.logsig = nn.Parameter(torch.full((M, in_dim), float(np.log(init_sigma))))
        self.w = nn.Parameter(torch.zeros(M))
        self._length_over_pi = nn.Parameter(
            torch.tensor(np.array(length_over_pi), dtype=torch.float32)[None, None, :],
            requires_grad=False,
        )
        self._periodic = nn.Parameter(
            torch.tensor(periodic, dtype=torch.bool)[None, None, :], requires_grad=False
        )

    def _delta2_fn(self, disp: torch.Tensor) -> torch.Tensor:
        return torch.where(
            self._periodic,
            self._length_over_pi * torch.sin(disp / self._length_over_pi),
            disp,
        ).square()

    def _delta2_grad(self, disp: torch.Tensor) -> torch.Tensor:
        return torch.where(
            self._periodic,
            self._length_over_pi * torch.sin(2 * disp / self._length_over_pi),
            2 * disp,
        )

    def _phi(self, disp: torch.Tensor) -> torch.Tensor:
        delta2 = self._delta2_fn(disp)
        sigma2 = torch.exp(2 * self.logsig)
        return torch.exp(-0.5 * (delta2 / sigma2[None, :, :]).sum(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        disp = x[:, None, :] - self.c[None, :, :]
        return self._phi(disp) @ self.w

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        disp = x[:, None, :] - self.c[None, :, :]
        Phi = self._phi(disp)
        sigma2 = torch.exp(2 * self.logsig)
        fac = (-0.5 * self.w[:, None] / sigma2)[None, :, :]
        return (fac * Phi[:, :, None] * self._delta2_grad(disp)).sum(1)


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
        x, G = batch
        return ((self.f.grad(x) - G) ** 2).mean()

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
        L = \frac{1}{N} \sum_{i=1}^N \left\| {\bf f}({\bf s}_i) - {\bf F}_i \right\|^2,

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
    patience
        The patience for early stopping.
    learning_rate
        The learning rate to be used for training.
    weight_decay
        The weight decay to be used for training.
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
        patience: int = 10,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
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
        self._patience = patience
        self._lr = learning_rate
        self._wd = weight_decay
        self._accelerator = accelerator
        self._num_workers = num_workers
        self._M = num_kernels

    @staticmethod
    def _as_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _create_dataloaders(
        self, X: torch.Tensor, G: torch.Tensor
    ) -> tuple[DataLoader, DataLoader | None]:
        N = len(X)
        idx = torch.randperm(N)
        nval = int(round(self._val_frac * N))
        val_idx, tr_idx = idx[:nval], idx[nval:]
        dl = DataLoader(
            TensorDataset(X[tr_idx], G[tr_idx]),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )
        if len(val_idx) == 0:
            return dl, None
        dlv = DataLoader(
            TensorDataset(X[val_idx], G[val_idx]),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )
        return dl, dlv

    def _create_callbacks(
        self, checkpoint_dir: str
    ) -> tuple[pl.callbacks.ModelCheckpoint, pl.callbacks.EarlyStopping]:
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_last=False,
            filename="best-{epoch:02d}-{val_loss:.6f}",
            dirpath=checkpoint_dir,
        )
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=self._patience
        )
        return checkpoint, early_stopping

    def fit(self, positions: ArrayLike, forces: ArrayLike, *, seed: int = 0) -> None:
        """Fit the potential regressor to the given positions and forces.

        Parameters
        ----------
        positions
            The positions to be used for training.
        forces
            The forces to be used for training.

        Keyword Arguments
        -----------------
        seed
            The seed to be used for training.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        X = torch.as_tensor(positions, dtype=torch.float32)
        G = -torch.as_tensor(forces, dtype=torch.float32)
        dl, dlv = self._create_dataloaders(X, G)

        self._model = GradMatch(
            dynamical_variables=self._dynamical_variables,
            M=self._M,
            lr=self._lr,
            wd=self._wd,
            init_sigma=self._init_sigma,
        )

        trainer_kwargs = {
            "max_epochs": self._num_epochs,
            "accelerator": self._accelerator,
            "devices": "auto",
            "logger": False,
            "enable_progress_bar": True,
        }
        if dlv is None:
            trainer_kwargs["enable_checkpointing"] = False
            trainer = pl.Trainer(**trainer_kwargs)
            trainer.fit(self._model, dl)
        else:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoint, early_stopping = self._create_callbacks(checkpoint_dir)
                trainer_kwargs["callbacks"] = [checkpoint, early_stopping]
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(self._model, dl, dlv)
                if checkpoint.best_model_path:
                    self._model = GradMatch.load_from_checkpoint(
                        checkpoint.best_model_path
                    )

    def predict(self, positions: ArrayLike) -> np.ndarray:
        """Predict the potential at the given positions.

        Parameters
        ----------
        positions
            An array of shape (M, d) containing the positions at which to predict the
            potential.

        Returns
        -------
        np.ndarray
            An array of shape (M,) containing the potential at the given positions.
        """
        device = next(self._model.parameters()).device
        X = torch.as_tensor(positions, dtype=torch.float32).to(device)
        with torch.no_grad():
            return self._as_numpy(self._model.f(X))

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
        centers = self._as_numpy(self._model.f.c)
        for dv, column in zip(self._dynamical_variables, centers.T):
            column[:], _ = np.vectorize(dv.bounds.wrap)(column, 0)
        return (
            centers,
            self._as_numpy(self._model.f.logsig.exp()),
            self._as_numpy(self._model.f.w),
        )
