"""
Unit tests for the openxps.regression module.
"""

import numpy as np
import pytest
import torch
from openmm import unit as mmunit

import openxps as xps
from openxps.bounds import NoBounds, PeriodicBounds
from openxps.regression import ForceMatchingRegressor, GradMatch, RBFPotential


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
        "r", mmunit.nanometer, mass_nm, NoBounds(0, 1, mmunit.dimensionless)
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

        # _model is None before fit, so accessing _model.f will raise AttributeError
        with pytest.raises((AttributeError, RuntimeError)):
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
