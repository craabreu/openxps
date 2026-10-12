"""
Unit tests for the openxps.regression module.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import pytorch as pl
from openmm import unit as mmunit

import openxps as xps
from openxps.bounds import NoBounds, PeriodicBounds
from openxps.regression import (
    ForceMatchingRegressor,
    GradMatch,
    RBFPotential,
    _LearningCurveRecorder,
)


def create_test_dynamical_variables():
    """Helper function to create test dynamical variables."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    dv1 = xps.DynamicalVariable(
        "phi", mmunit.radian, mass, PeriodicBounds(-np.pi, np.pi, mmunit.radian)
    )
    dv2 = xps.DynamicalVariable(
        "psi", mmunit.radian, mass, PeriodicBounds(-np.pi, np.pi, mmunit.radian)
    )
    return [dv1, dv2]


def create_test_dynamical_variables_mixed():
    """Helper function to create mixed periodic/non-periodic variables."""
    mass_rad = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    mass_nm = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.nanometer) ** 2
    dv1 = xps.DynamicalVariable(
        "phi", mmunit.radian, mass_rad, PeriodicBounds(-np.pi, np.pi, mmunit.radian)
    )
    dv2 = xps.DynamicalVariable(
        "r", mmunit.nanometer, mass_nm, NoBounds(0, 1, mmunit.nanometer)
    )
    return [dv1, dv2]


class TestRBFPotential:
    """Tests for RBFPotential class."""

    def test_initialization(self):
        """Test RBFPotential initialization."""
        dvs = create_test_dynamical_variables()
        potential = RBFPotential(dvs, M=10)

        assert potential.c.shape == (10, 2)
        assert potential.logsig.shape == (10, 2)
        assert potential.w.shape == (10,)
        assert potential._length_over_pi.shape == (1, 1, 2)
        assert potential._periodic.shape == (1, 1, 2)

    def test_forward(self):
        """Test forward pass of RBFPotential."""
        dvs = create_test_dynamical_variables()
        potential = RBFPotential(dvs, M=5)
        x = torch.randn(3, 2)
        output = potential(x)

        assert output.shape == (3,)
        assert torch.all(torch.isfinite(output))

    def test_grad(self):
        """Test gradient computation of RBFPotential."""
        dvs = create_test_dynamical_variables()
        potential = RBFPotential(dvs, M=5)
        x = torch.randn(3, 2, requires_grad=True)
        grad_output = potential.grad(x)

        assert grad_output.shape == (3, 2)
        assert torch.all(torch.isfinite(grad_output))

    def test_periodic_distance(self):
        """Test periodic distance function."""
        dvs = create_test_dynamical_variables()
        potential = RBFPotential(dvs, M=5)
        disp = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        delta2 = potential._delta2_fn(disp)

        assert delta2.shape == (1, 1, 2)
        assert torch.all(delta2 >= 0)  # Squared distance should be non-negative

    def test_non_periodic_distance(self):
        """Test non-periodic distance function."""
        dvs = create_test_dynamical_variables_mixed()
        potential = RBFPotential(dvs, M=5)
        disp = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        delta2 = potential._delta2_fn(disp)

        assert delta2.shape == (1, 1, 2)
        assert torch.all(delta2 >= 0)

    def test_mixed_periodic_non_periodic(self):
        """Test potential with mixed periodic and non-periodic variables."""
        dvs = create_test_dynamical_variables_mixed()
        potential = RBFPotential(dvs, M=5)
        x = torch.randn(3, 2)
        output = potential(x)
        grad_output = potential.grad(x)

        assert output.shape == (3,)
        assert grad_output.shape == (3, 2)
        assert torch.all(torch.isfinite(output))
        assert torch.all(torch.isfinite(grad_output))

    def test_learn_centers_false(self):
        """Test that centers are not learnable when learn_centers=False."""
        dvs = create_test_dynamical_variables()
        potential = RBFPotential(dvs, M=5, learn_centers=False)

        assert not potential.c.requires_grad

    def test_centers_start_inside_bounds(self):
        """Centers are drawn uniformly in [lower, upper]."""
        dvs = create_test_dynamical_variables_mixed()
        torch.manual_seed(0)
        potential = RBFPotential(dvs, M=2000)
        centers = potential.c.detach().numpy()
        for column, dv in zip(centers.T, dvs):
            bounds = dv.bounds
            assert column.min() >= bounds.lower
            assert column.max() <= bounds.upper
            midpoint = 0.5 * (bounds.lower + bounds.upper)
            assert abs(column.mean() - midpoint) < 0.05 * bounds.length

    def test_constants_are_buffers(self):
        potential = RBFPotential(create_test_dynamical_variables(), M=5)
        assert {name for name, _ in potential.named_buffers()} == {
            "_length_over_pi",
            "_periodic",
        }
        assert {name for name, _ in potential.named_parameters()} == {
            "c",
            "logsig",
            "w",
        }


class TestGradMatch:
    """Tests for GradMatch class."""

    def test_initialization(self):
        """Test GradMatch initialization."""
        dvs = create_test_dynamical_variables()
        model = GradMatch(dvs, M=10, lr=1e-3, wd=1e-4)

        assert model.f is not None
        assert isinstance(model.f, RBFPotential)

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        dvs = create_test_dynamical_variables()
        model = GradMatch(dvs, M=5, lr=1e-3, wd=1e-4)
        optimizer = model.configure_optimizers()

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    def test_training_step(self):
        """Test training step."""
        dvs = create_test_dynamical_variables()
        model = GradMatch(dvs, M=5, lr=1e-3)
        x = torch.randn(4, 2)
        G = torch.randn(4, 2)
        batch = (x, G)
        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_validation_step(self):
        """Test validation step."""
        dvs = create_test_dynamical_variables()
        model = GradMatch(dvs, M=5, lr=1e-3)
        x = torch.randn(4, 2)
        G = torch.randn(4, 2)
        batch = (x, G)
        loss = model.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_loss_computation(self):
        """Test loss computation."""
        dvs = create_test_dynamical_variables()
        model = GradMatch(dvs, M=5, lr=1e-3)
        x = torch.randn(4, 2)
        G = torch.randn(4, 2)
        batch = (x, G)
        loss = model._loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_hyperparameters_exclude_dynamical_variables(self):
        """Checkpoints must be loadable with weights_only=True."""
        model = GradMatch(create_test_dynamical_variables(), M=4)
        assert set(model.hparams) == {"M", "lr", "wd"}

    def test_weight_decay_applies_only_to_weights(self):
        model = GradMatch(create_test_dynamical_variables(), M=4)
        optimizer = model.configure_optimizers()
        decay = {
            id(param): group["weight_decay"]
            for group in optimizer.param_groups
            for param in group["params"]
        }
        assert decay[id(model.f.w)] == pytest.approx(1e-4)
        assert decay[id(model.f.c)] == 0.0
        assert decay[id(model.f.logsig)] == 0.0


class TestForceMatchingRegressor:
    """Tests for ForceMatchingRegressor class."""

    def test_initialization(self):
        """Test ForceMatchingRegressor initialization."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=10)

        assert regressor._M == 10
        assert regressor._num_epochs == 50
        assert regressor._batch_size == 256
        assert len(regressor._dynamical_variables) == 2

    def test_fit_basic(self):
        """Test basic fit functionality."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=2, validation_fraction=0.2
        )

        # Generate synthetic data
        np.random.seed(42)
        N = 100
        positions = np.random.randn(N, 2) * 0.5
        forces = -np.random.randn(N, 2) * 0.1

        regressor.fit(positions, forces)
        assert regressor._model is not None

    def test_fit_no_validation(self):
        """Test fit with no validation set."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=2, validation_fraction=0.0
        )

        np.random.seed(42)
        N = 50
        positions = np.random.randn(N, 2) * 0.5
        forces = -np.random.randn(N, 2) * 0.1

        regressor.fit(positions, forces)
        assert regressor._model is not None

    def test_predict(self):
        """Test prediction functionality."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=2, validation_fraction=0.2
        )

        np.random.seed(42)
        N_train = 50
        positions_train = np.random.randn(N_train, 2) * 0.5
        forces_train = -np.random.randn(N_train, 2) * 0.1

        regressor.fit(positions_train, forces_train)

        # Test prediction
        N_test = 10
        positions_test = np.random.randn(N_test, 2) * 0.5
        predictions = regressor.predict(positions_test)

        assert predictions.shape == (N_test,)
        assert np.all(np.isfinite(predictions))

    def test_get_parameters(self):
        """Test get_parameters method."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=2, validation_fraction=0.2
        )

        np.random.seed(42)
        N = 50
        positions = np.random.randn(N, 2) * 0.5
        forces = -np.random.randn(N, 2) * 0.1

        regressor.fit(positions, forces)
        centers, sigmas, weights = regressor.get_parameters()

        assert centers.shape == (5, 2)
        assert sigmas.shape == (5, 2)
        assert weights.shape == (5,)
        assert np.all(np.isfinite(centers))
        assert np.all(np.isfinite(sigmas))
        assert np.all(np.isfinite(weights))
        assert np.all(sigmas > 0)  # Sigmas should be positive

    def test_get_parameters_wrapping(self):
        """Test that centers are wrapped according to bounds."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=2, validation_fraction=0.2
        )

        np.random.seed(42)
        N = 50
        positions = np.random.randn(N, 2) * 0.5
        forces = -np.random.randn(N, 2) * 0.1

        regressor.fit(positions, forces)
        centers, sigmas, weights = regressor.get_parameters()

        # Check that periodic centers are within bounds
        for i, dv in enumerate(regressor._dynamical_variables):
            if isinstance(dv.bounds, PeriodicBounds):
                lower = dv.bounds.lower
                upper = dv.bounds.upper
                assert np.all(centers[:, i] >= lower)
                assert np.all(centers[:, i] <= upper)

    def test_mixed_periodic_non_periodic(self):
        """Test regressor with mixed periodic and non-periodic variables."""
        dvs = create_test_dynamical_variables_mixed()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=2, validation_fraction=0.2
        )

        np.random.seed(42)
        N = 50
        positions = np.random.randn(N, 2) * 0.5
        forces = -np.random.randn(N, 2) * 0.1

        regressor.fit(positions, forces)
        predictions = regressor.predict(positions[:5])
        centers, sigmas, weights = regressor.get_parameters()

        assert predictions.shape == (5,)
        assert centers.shape == (5, 2)
        assert sigmas.shape == (5, 2)
        assert weights.shape == (5,)

    def test_custom_parameters(self):
        """Test regressor with custom parameters."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs,
            num_kernels=10,
            validation_fraction=0.15,
            batch_size=32,
            num_epochs=3,
            learning_rate=1e-3,
            weight_decay=1e-5,
            patience=5,
        )

        assert regressor._M == 10
        assert regressor._val_frac == 0.15
        assert regressor._batch_size == 32
        assert regressor._num_epochs == 3
        assert regressor._lr == 1e-3
        assert regressor._wd == 1e-5
        assert regressor._patience == 5

    def test_predict_before_fit(self):
        """Test that predict raises error if called before fit."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5)

        positions = np.random.randn(5, 2)
        # _model is None before fit, so accessing _model.f will raise AttributeError
        with pytest.raises((AttributeError, RuntimeError)):
            regressor.predict(positions)

    def test_get_parameters_before_fit(self):
        """Test that get_parameters raises error if called before fit."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5)

        with pytest.raises(RuntimeError, match="not been fitted"):
            regressor.get_parameters()

    def test_early_stopping_with_validation(self):
        """Test that early stopping is used when validation set exists."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=100, validation_fraction=0.2, patience=2
        )

        np.random.seed(42)
        N = 50
        positions = np.random.randn(N, 2) * 0.5
        forces = -np.random.randn(N, 2) * 0.1

        # This should complete without error (may stop early)
        regressor.fit(positions, forces)
        assert regressor._model is not None

    def test_checkpoint_loading(self):
        """Test that best model checkpoint is loaded."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=5, validation_fraction=0.2
        )

        np.random.seed(42)
        N = 50
        positions = np.random.randn(N, 2) * 0.5
        forces = -np.random.randn(N, 2) * 0.1

        regressor.fit(positions, forces)
        # Model should be loaded from checkpoint if validation set exists
        assert regressor._model is not None
        predictions = regressor.predict(positions[:5])
        assert predictions.shape == (5,)


class TestFitBehavior:
    def test_fit_with_validation_records_epochs(self):
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=3, validation_fraction=0.2
        )
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 0.5
        forces = -rng.standard_normal((100, 2)) * 0.1
        regressor.fit(positions, forces, seed=1)
        curve = regressor.get_learning_curve()
        assert list(curve.epoch) == [0, 1, 2]
        assert {"train_loss", "val_loss"} <= set(curve.columns)
        assert np.all(np.isfinite(regressor.predict(positions)))

    @pytest.mark.parametrize(
        ("num_samples", "fraction"), [(2, 0.75), (3, 0.9), (0, 0.0)]
    )
    def test_fit_rejects_split_without_training_data(self, num_samples, fraction):
        regressor = ForceMatchingRegressor(
            create_test_dynamical_variables(), 5, validation_fraction=fraction
        )
        positions = np.zeros((num_samples, 2))
        with pytest.raises(ValueError, match="No training data"):
            regressor.fit(positions, positions)
        with pytest.raises(RuntimeError, match="not been fitted"):
            regressor.get_learning_curve()

    def test_rejects_validation_fraction_without_training_data(self):
        with pytest.raises(ValueError, match="validation_fraction"):
            ForceMatchingRegressor(
                create_test_dynamical_variables(), 5, validation_fraction=1.0
            )

    def test_devices_default_to_one(self):
        regressor = ForceMatchingRegressor(
            create_test_dynamical_variables(), num_kernels=5
        )
        assert regressor._devices == 1

    def test_fit_rejects_mismatched_shapes(self):
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5, num_epochs=1)
        with pytest.raises(ValueError, match=r"positions must have shape \(N, 2\)"):
            regressor.fit(np.zeros((10, 3)), np.zeros((10, 3)))
        with pytest.raises(ValueError, match="forces must have the same shape"):
            regressor.fit(np.zeros((10, 2)), np.zeros(10))

    def test_fit_twice_resets_model_and_curve(self):
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=2, validation_fraction=0.0
        )
        rng = np.random.default_rng(0)
        positions = rng.standard_normal((40, 2)) * 0.5
        forces = -rng.standard_normal((40, 2)) * 0.1
        with pytest.raises(RuntimeError, match="not been fitted"):
            regressor.get_learning_curve()
        regressor.fit(positions, forces, seed=1)
        first_centers, _, _ = regressor.get_parameters()
        first_curve = regressor.get_learning_curve()
        regressor._num_epochs = 1
        regressor.fit(positions, forces, seed=2)
        second_centers, _, _ = regressor.get_parameters()
        assert len(regressor.get_learning_curve()) == 1
        assert len(first_curve) == 2
        assert not np.allclose(first_centers, second_centers)

    def test_get_parameters_does_not_change_predictions(self):
        mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
        dvs = [
            xps.DynamicalVariable(
                "phi", mmunit.radian, mass, PeriodicBounds(-np.pi, np.pi, mmunit.radian)
            ),
            xps.DynamicalVariable(
                "r",
                mmunit.nanometer,
                3 * mmunit.dalton,
                xps.ReflectiveBounds(0, 1, mmunit.nanometer),
            ),
        ]
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=20, num_epochs=1, validation_fraction=0.0
        )
        rng = np.random.default_rng(0)
        positions = np.column_stack(
            [rng.uniform(-np.pi, np.pi, 50), rng.uniform(0, 1, 50)]
        )
        regressor.fit(positions, -positions, seed=1)
        with torch.no_grad():
            regressor._model.f.c += 1.5
        before = regressor.predict(positions)
        centers, sigmas, weights = regressor.get_parameters()
        np.testing.assert_allclose(regressor.predict(positions), before)

        # Periodic centers are wrapped; reflective ones are returned as fitted
        assert np.all((centers[:, 0] >= -np.pi) & (centers[:, 0] < np.pi))
        np.testing.assert_allclose(
            centers[:, 1], regressor._model.f.c[:, 1].detach().numpy()
        )
        assert np.any(centers[:, 1] > 1)

        # The returned parameters reproduce the fitted potential
        disp = positions[:, None, :] - centers[None, :, :]
        period = 2 * np.pi
        delta = np.stack(
            [period / np.pi * np.sin(np.pi * disp[..., 0] / period), disp[..., 1]],
            axis=-1,
        )
        rebuilt = np.exp(-0.5 * (delta**2 / sigmas[None] ** 2).sum(-1)) @ weights
        np.testing.assert_allclose(rebuilt, before, rtol=1e-4, atol=1e-6)

    def test_fit_does_not_touch_global_numpy_rng(self):
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=1, validation_fraction=0.0
        )
        np.random.seed(123)
        expected = np.random.rand()
        np.random.seed(123)
        regressor.fit(np.zeros((20, 2)), np.zeros((20, 2)), seed=None)
        assert np.random.rand() == expected

    @pytest.mark.parametrize("seed", [None, 7])
    @pytest.mark.parametrize("validation_fraction", [0.0, 0.2])
    def test_fit_does_not_touch_global_torch_rng(self, seed, validation_fraction):
        regressor = ForceMatchingRegressor(
            create_test_dynamical_variables(),
            num_kernels=5,
            num_epochs=2,
            batch_size=4,
            validation_fraction=validation_fraction,
        )
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        regressor.fit(np.zeros((20, 2)), np.zeros((20, 2)), seed=seed)
        torch.testing.assert_close(torch.rand(3), expected)

    def test_same_seed_reproduces_fit(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(-np.pi, np.pi, (40, 2))
        forces = -np.sin(positions)
        results = []
        for global_seed in (1, 2):
            torch.manual_seed(global_seed)
            regressor = ForceMatchingRegressor(
                create_test_dynamical_variables(),
                num_kernels=5,
                num_epochs=3,
                batch_size=8,
                validation_fraction=0.2,
            )
            regressor.fit(positions, forces, seed=11)
            results.append(regressor.get_parameters())
        for first, second in zip(*results):
            np.testing.assert_array_equal(first, second)

    def test_failed_fit_keeps_previous_model(self, monkeypatch):
        regressor = ForceMatchingRegressor(
            create_test_dynamical_variables(),
            num_kernels=5,
            num_epochs=1,
            validation_fraction=0.0,
        )
        positions = np.random.default_rng(0).uniform(-1, 1, (20, 2))
        regressor.fit(positions, -positions, seed=1)
        model = regressor._model
        curve = regressor.get_learning_curve()
        before = regressor.predict(positions)

        def fail(*args, **kwargs):
            raise RuntimeError("training failed")

        monkeypatch.setattr(pl.Trainer, "fit", fail)
        with pytest.raises(RuntimeError, match="training failed"):
            regressor.fit(positions, positions, seed=2)
        assert regressor._model is model
        pd.testing.assert_frame_equal(regressor.get_learning_curve(), curve)
        np.testing.assert_array_equal(regressor.predict(positions), before)

    @pytest.mark.parametrize("shape", [(10,), (10, 3), (2, 5, 2)])
    def test_predict_validates_shape(self, shape):
        regressor = ForceMatchingRegressor(
            create_test_dynamical_variables(),
            num_kernels=5,
            num_epochs=1,
            validation_fraction=0.0,
        )
        regressor.fit(np.zeros((20, 2)), np.zeros((20, 2)), seed=1)
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            regressor.predict(np.zeros(shape))

    def test_fit_emits_no_checkpoint_directory_warning(self):
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(
            dvs, num_kernels=5, num_epochs=1, validation_fraction=0.2
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regressor.fit(np.zeros((20, 2)), np.zeros((20, 2)), seed=1)
        assert not [w for w in caught if "exists and is not empty" in str(w.message)]


class _FakeTrainer:
    def __init__(self, epoch, metrics, sanity_checking=False):
        self.current_epoch = epoch
        self.callback_metrics = metrics
        self.sanity_checking = sanity_checking


def test_learning_curve_recorder_tolerates_nan_validation_loss():
    """A NaN val_loss must not break the epoch alignment."""
    recorder = _LearningCurveRecorder()
    sanity = _FakeTrainer(0, {"val_loss": torch.tensor(9.0)}, sanity_checking=True)
    recorder.on_validation_epoch_end(sanity, None)
    for epoch, val in enumerate((0.5, float("nan"), 0.3)):
        trainer = _FakeTrainer(epoch, {"val_loss": torch.tensor(val)})
        recorder.on_validation_epoch_end(trainer, None)
        trainer = _FakeTrainer(epoch, {"train_loss": torch.tensor(1.0 / (epoch + 1))})
        recorder.on_train_epoch_end(trainer, None)
    frame = recorder.to_frame()
    assert list(frame.columns) == ["epoch", "val_loss", "train_loss"]
    assert list(frame.epoch) == [0, 1, 2]
    assert np.isnan(frame.val_loss[1])
    assert frame.train_loss[2] == pytest.approx(1 / 3)
