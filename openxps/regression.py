"""
.. module:: openxps.rbf
   :platform: Linux, MacOS, Windows
   :synopsis: Radial basis function regressor for free energy surface fitting.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import tempfile
import typing as t

import numpy as np
import pandas as pd
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


INIT_NUM_SIGMAS = 4.0


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
    learn_centers
        Whether to learn the centers of the radial basis functions.
    generator
        The random number generator used to place the initial centers. If None,
        PyTorch's global generator is used.
    """

    def __init__(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        M: int = 256,
        learn_centers: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        lengths, periodic, random_points = [], [], []
        for dv in dynamical_variables:
            lengths.append(dv.bounds.length)
            periodic.append(isinstance(dv.bounds, PeriodicBounds))
            random_points.append(
                torch.rand(M, generator=generator) * dv.bounds.length + dv.bounds.lower
            )

        self.c = nn.Parameter(
            torch.stack(random_points, dim=-1), requires_grad=learn_centers
        )
        log_sigmas = np.log(lengths) - np.log(INIT_NUM_SIGMAS)
        self.logsig = nn.Parameter(
            torch.tensor(np.repeat(log_sigmas[None, :], M, axis=0), dtype=torch.float32)
        )
        self.w = nn.Parameter(torch.zeros(M))
        self.register_buffer(
            "_length_over_pi",
            torch.tensor(np.array(lengths) / np.pi, dtype=torch.float32)[None, None, :],
        )
        self.register_buffer(
            "_periodic", torch.tensor(periodic, dtype=torch.bool)[None, None, :]
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
    """Gradient matching regressor.

    Parameters
    ----------
    dynamical_variables
        A sequence of dynamical variables defining the dimensions and bounds
        for the potential. The length of this sequence determines the input
        dimension.

    Keyword Arguments
    -----------------
    M
        The number of radial basis functions.
    lr
        The learning rate.
    wd
        The weight decay.
    generator
        The random number generator used to place the initial kernel centers.
    """

    def __init__(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        M: int = 256,
        lr: float = 2e-3,
        wd: float = 1e-4,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dynamical_variables", "generator"])
        self.f = RBFPotential(dynamical_variables, M, generator=generator)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {"params": [self.f.w], "weight_decay": self.hparams.wd},
                {"params": [self.f.c, self.f.logsig], "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )

    def _loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, G = batch
        return ((self.f.grad(x) - G) ** 2).mean()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


class _LearningCurveRecorder(pl.Callback):
    """Collect per-epoch training and validation losses keyed by epoch."""

    def __init__(self) -> None:
        super().__init__()
        self._records: dict[int, dict[str, float]] = {}

    def _record(self, trainer: pl.Trainer, key: str) -> None:
        value = trainer.callback_metrics.get(key)
        if value is not None:
            epoch = trainer.current_epoch
            record = self._records.setdefault(epoch, {"epoch": epoch})
            record[key] = float(value)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._record(trainer, "train_loss")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.sanity_checking:
            self._record(trainer, "val_loss")

    def to_frame(self) -> pd.DataFrame:
        """Return the recorded losses with one row per epoch."""
        return pd.DataFrame(list(self._records.values()))


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
        The dynamical variables to be used in the potential. They are converted to
        the MD unit system, so the positions and forces passed to :meth:`fit` and
        :meth:`predict` must be expressed in MD units.
    num_kernels
        The number of kernels to be used in the potential.

    Keyword Arguments
    -----------------
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
    devices
        The number of devices, or ``"auto"``, passed to the Lightning trainer. The
        default of 1 avoids launching distributed training on multi-GPU hosts.
    num_workers
        The number of data loading workers to be used for training.
    """

    def __init__(  # noqa: PLR0913
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        num_kernels: int,
        *,
        validation_fraction: float = 0.1,
        batch_size: int = 256,
        num_epochs: int = 50,
        patience: int = 10,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        accelerator: str = "auto",
        devices: int | str = 1,
        num_workers: int = 0,
    ):
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in the interval [0, 1).")
        self._dynamical_variables = tuple(
            dv.in_md_units() for dv in dynamical_variables
        )
        self._val_frac = validation_fraction
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._patience = patience
        self._lr = learning_rate
        self._wd = weight_decay
        self._accelerator = accelerator
        self._devices = devices
        self._num_workers = num_workers
        self._M = num_kernels
        self._model = None
        self._metrics = None

    @staticmethod
    def _as_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().clone().numpy()

    def _num_validation_samples(self, num_samples: int) -> int:
        return int(round(self._val_frac * num_samples))

    def _create_dataloaders(
        self, X: torch.Tensor, G: torch.Tensor, generator: torch.Generator
    ) -> tuple[DataLoader, DataLoader | None]:
        N = len(X)
        idx = torch.randperm(N, generator=generator)
        nval = self._num_validation_samples(N)
        val_idx, tr_idx = idx[:nval], idx[nval:]
        train_dl = DataLoader(
            TensorDataset(X[tr_idx], G[tr_idx]),
            batch_size=self._batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
        )
        if len(val_idx) == 0:
            return train_dl, None
        val_dl = DataLoader(
            TensorDataset(X[val_idx], G[val_idx]),
            batch_size=self._batch_size,
            generator=generator,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
        )
        return train_dl, val_dl

    def _create_callbacks(
        self, directory: str
    ) -> tuple[pl.callbacks.ModelCheckpoint, pl.callbacks.EarlyStopping]:
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_last=False,
            filename="best-{epoch:02d}-{val_loss:.6f}",
            dirpath=directory,
        )
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=self._patience
        )
        return checkpoint, early_stopping

    def _as_positions(self, positions: ArrayLike) -> torch.Tensor:
        X = torch.as_tensor(np.asarray(positions, dtype=np.float32))
        d = len(self._dynamical_variables)
        if X.ndim != 2 or X.shape[1] != d:  # noqa: PLR2004
            raise ValueError(
                f"positions must have shape (N, {d}), got {tuple(X.shape)}"
            )
        return X

    def _validate_forces(self, X: torch.Tensor, G: torch.Tensor) -> None:
        if G.shape != X.shape:
            raise ValueError(
                f"forces must have the same shape as positions {tuple(X.shape)}, "
                f"got {tuple(G.shape)}"
            )

    def fit(
        self, positions: ArrayLike, forces: ArrayLike, *, seed: int | None = None
    ) -> None:
        """Fit the potential regressor to the given positions and forces.

        Parameters
        ----------
        positions
            An array of shape (N, d) with the positions in MD units.
        forces
            An array of shape (N, d) with the forces in MD units.

        Keyword Arguments
        -----------------
        seed
            The seed used to split the data, shuffle the batches and initialize the
            kernels. If None, a random seed is used. The global random states of
            NumPy and PyTorch are left untouched.

        Notes
        -----
        The fitted model and learning curve replace the previous ones only if
        training completes, so a failed fit leaves the regressor unchanged.
        """
        generator = torch.Generator()
        generator.manual_seed(int(np.random.default_rng(seed).integers(0, 2**63 - 1)))

        X = self._as_positions(positions)
        G = -torch.as_tensor(np.asarray(forces, dtype=np.float32))
        self._validate_forces(X, G)
        num_samples = len(X)
        if num_samples - self._num_validation_samples(num_samples) < 1:
            raise ValueError(
                f"No training data left: {num_samples} samples with "
                f"validation_fraction={self._val_frac}."
            )
        train_dl, val_dl = self._create_dataloaders(X, G, generator)

        model = GradMatch(
            self._dynamical_variables, self._M, self._lr, self._wd, generator
        )
        recorder = _LearningCurveRecorder()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer_kwargs = {
                "max_epochs": self._num_epochs,
                "accelerator": self._accelerator,
                "devices": self._devices,
                "logger": False,
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "callbacks": [recorder],
            }
            if val_dl is None:
                trainer_kwargs["enable_checkpointing"] = False
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(model, train_dl)
            else:
                checkpoint, early_stopping = self._create_callbacks(tmp_dir)
                trainer_kwargs["callbacks"] += [checkpoint, early_stopping]
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(model, train_dl, val_dl)
                if checkpoint.best_model_path:
                    state = torch.load(
                        checkpoint.best_model_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    model.load_state_dict(state["state_dict"])

        self._model = model
        self._metrics = recorder.to_frame()

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
        if self._model is None:
            raise RuntimeError("The model has not been fitted yet.")
        device = next(self._model.parameters()).device
        X = self._as_positions(positions).to(device)
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

        Notes
        -----
        Centers of periodic variables are wrapped into their bounds. Centers of
        non-periodic variables are returned as fitted, even if they lie outside the
        bounds, because the kernels use plain displacements for those variables.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._model is None:
            raise RuntimeError("The model has not been fitted yet.")
        centers = self._as_numpy(self._model.f.c)
        for dv, column in zip(self._dynamical_variables, centers.T):
            if isinstance(dv.bounds, PeriodicBounds):
                column[:], _ = np.vectorize(dv.bounds.wrap)(column, 0)
        return (
            centers,
            self._as_numpy(self._model.f.logsig.exp()),
            self._as_numpy(self._model.f.w),
        )

    def get_learning_curve(self) -> pd.DataFrame:
        """Get the learning curve of the last fit.

        Returns
        -------
        pd.DataFrame
            A copy of the per-epoch losses, with columns ``epoch``, ``train_loss``
            and, when a validation set is used, ``val_loss``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._metrics is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self._metrics.copy()
