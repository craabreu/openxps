import os
import pickle
import re
import warnings
from copy import copy

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app.metadynamics import _LoadedBias

from openmm.app import StateDataReporter

from .online_kde import CVSpace, OnlineKDE

STATS_WINDOW_SIZE = 10
CORRECTED_OPES_EXPLORE = True
USE_PDF_OPES_EXPLORE = True


class RunningAverage:
    """Class to handle running average calculations."""

    def __init__(self, numDimensions=1):
        self._num = 0
        self._total = np.zeros(numDimensions)

    def __getstate__(self):
        return {"num": self._num, "total": self._total}

    def __setstate__(self, state):
        self._num = state["num"]
        self._total = state["total"]

    def __iadd__(self, other):
        self._total += other._total
        self._num += other._num
        return self

    def copy(self):
        new = self.__class__(self._total.shape[0])
        new._num = self._num
        new._total = self._total
        return new

    def update(self, sample):
        """Update the running average with a new sample."""
        self._num += 1
        self._total += sample

    def get(self):
        """Get the running average."""
        return self._total / max(1, self._num)


class OPES:
    """Performs On-the-fly Probability Enhanced Sampling (OPES).

    This class implements the On-the-fly Probability Enhanced Sampling (OPES) method,
    as described in two papers by Invernizzi and Parrinello:

    1. Rethinking Metadynamics: From Bias Potentials to Probability Distributions
       https://doi.org/10.1021/acs.jpclett.0c00497

    2. Exploration vs Convergence Speed in Adaptive-Bias Enhanced Sampling
       https://doi.org/10.1021/acs.jctc.2c00152

    Parameters
    ----------
    system
        The System to simulate. A CustomCVForce implementing the bias is created and
        added to the System.
    variables
        The collective variables whose sampling should be enhanced.
    temperature
        The temperature at which the simulation will be run.
    barrier
        The energy barrier that the simulation should overcome.
    frequency
        The interval in time steps at which to add a new kernel to the probability
        distribution estimate.
    varianceFrequency
        The interval in time steps at which to update the variance of the collective
        variables.
    biasFactor
        The bias factor to use. If None, then barrier / kT is used.
    exploreMode
        Whether to apply the OPES-Explore variant.
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier,
        frequency,
        varianceFrequency,
        biasFactor=None,
        exploreMode=False,
        stateIDFuncs=(lambda _: True,),
        bounded=False,
        saveFrequency=None,
        biasDir=None,
    ):
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole
        self.variables = variables
        self.temperature = temperature
        self.barrier = barrier
        self.frequency = frequency
        self.varianceFrequency = varianceFrequency
        self.biasFactor = biasFactor
        self.exploreMode = exploreMode
        self.stateIDFuncs = stateIDFuncs
        self.bounded = bounded
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir

        d = len(variables)
        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        biasFactor = barrier / kbt if biasFactor is None else biasFactor
        numPeriodics = sum(v.periodic for v in variables)
        freeGroups = set(range(32)) - set(f.getForceGroup() for f in system.getForces())
        self._validate(d, biasFactor, numPeriodics, freeGroups)

        prefactor = (1 - 1 / biasFactor) * kbt
        if exploreMode:
            prefactor *= biasFactor
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._biasFactor = biasFactor
        self._prefactor = prefactor
        self._logEpsilon = -barrier / prefactor

        self._cvSpace = CVSpace(variables, bounded)
        self._cases = ("total",) + ("self",) * bool(saveFrequency)
        self._kde = {}
        numLabels = len(self.stateIDFuncs)
        for case in self._cases:
            self._kde[case] = OnlineKDE(self._cvSpace, numLabels)
            self._kde[f"{case}.rw"] = OnlineKDE(self._cvSpace, numLabels)

        self._adaptiveVariance = varianceFrequency is not None
        self._interval = varianceFrequency or frequency
        self._variance = {case: RunningAverage(d) for case in self._cases}
        if self._adaptiveVariance:
            self._tau = STATS_WINDOW_SIZE * frequency // varianceFrequency
            self._counter = 0
            self._sampleMean = np.zeros(d)
        else:
            sqdev = np.array([cv.biasWidth**2 for cv in variables])
            for case in self._cases:
                self._variance[case].update(sqdev)

        self._lastVisitedState = 0

        if saveFrequency:
            self._id = np.random.RandomState().randint(0x7FFFFFFF)
            self._saveIndex = 0
            self._loadedBiases = {}
            self._syncWithDisk()

        gridWidths = [cv.gridWidth for cv in variables]
        self._widths = [] if d == 1 else gridWidths
        self._limits = sum(([cv.minValue, cv.maxValue] for cv in variables), [])

        energyFunction = "table(" + ",".join(f"cv{i}" for i in range(d)) + ")"
        self._force = mm.CustomCVForce(energyFunction)
        for i, var in enumerate(variables):
            self._force.addCollectiveVariable(f"cv{i}", var.force)
        table = getattr(mm, f"Continuous{d}DFunction")(
            *self._widths,
            np.full(np.prod(gridWidths), -barrier / unit.kilojoules_per_mole),
            *self._limits,
            numPeriodics == d,
        )
        self._force.addTabulatedFunction("table", table)
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def _validate(self, d, biasFactor, numPeriodics, freeGroups):
        if self.varianceFrequency and (self.frequency % self.varianceFrequency != 0):
            raise ValueError("varianceFrequency must be a divisor of frequency")
        if (self.saveFrequency is None) != (self.biasDir is None):
            raise ValueError("Must specify both saveFrequency and biasDir")
        if self.saveFrequency and (self.saveFrequency % self.frequency != 0):
            raise ValueError("saveFrequency must be a multiple of frequency")
        if biasFactor <= 1.0:
            raise ValueError("OPES barrier must be greater than 1 kT")
        if numPeriodics not in [0, d]:
            raise ValueError("OPES cannot handle mixed periodic/non-periodic variables")
        if not 1 <= d <= 3:
            raise ValueError("OPES requires 1, 2, or 3 collective variables")
        if not freeGroups:
            raise RuntimeError("OPES requires a free force group, but all are in use.")
        if not all(map(callable, self.stateIDFuncs)):
            raise ValueError("stateIDFuncs must be a list of functions")

    def _updateSampleStats(self, values):
        """Update the sample mean and variance of the collective variables."""
        self._counter += 1
        delta = self._cvSpace.displacement(self._sampleMean, values)
        x = 1 / min(self._tau, self._counter)
        self._sampleMean = self._cvSpace.endpoint(self._sampleMean, x * delta)
        sqdev = delta * self._cvSpace.displacement(self._sampleMean, values)
        for case in self._cases:
            self._variance[case].update(sqdev)

    def _syncWithDisk(self):
        """
        Save biases to disk, and check for updated files created by other processes.
        """

        oldName = os.path.join(self.biasDir, f"kde_{self._id}_{self._saveIndex}.pkl")
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, f"temp_{self._id}_{self._saveIndex}.pkl")
        fileName = os.path.join(self.biasDir, f"kde_{self._id}_{self._saveIndex}.pkl")
        data = {
            "kde": self._kde["self"],
            "kde.rw": self._kde["self.rw"],
            "var": self._variance["self"],
        }
        with open(tempName, "wb") as file:
            pickle.dump(data, file)
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)

        fileLoaded = False
        pattern = re.compile(r"kde_(.*)_(.*)\.pkl")
        for filename in os.listdir(self.biasDir):
            match = pattern.match(filename)
            if match is not None:
                matchId = int(match.group(1))
                matchIndex = int(match.group(2))
                if matchId != self._id and (
                    matchId not in self._loadedBiases
                    or matchIndex > self._loadedBiases[matchId].index
                ):
                    try:
                        with open(os.path.join(self.biasDir, filename), "rb") as file:
                            data = pickle.load(file)
                        self._loadedBiases[matchId] = _LoadedBias(
                            matchId, matchIndex, data
                        )
                        fileLoaded = True
                    except IOError:
                        warnings.warn(
                            f"The file {filename} seems to have been deleted. Using"
                            "the lastest loaded data from the same process."
                        )

        if fileLoaded:
            self._kde["total"] = copy(self._kde["self"])
            self._variance["total"] = copy(self._variance["self"])
            self._kde["total.rw"] = copy(self._kde["self.rw"])
            for bias in self._loadedBiases.values():
                self._kde["total"] += bias.bias["kde"]
                self._variance["total"] += bias.bias["var"]
                self._kde["total.rw"] += bias.bias["kde.rw"]

    def getNumKernels(self):
        """Get the number of kernels in the kernel density estimator."""
        kde = self._kde["total" if self.exploreMode else "total.rw"]
        return kde.getNumKernels()

    def getBias(self):
        """Get the OPES bias potential evaluated on the grid."""
        kde = self._kde["total" if self.exploreMode else "total.rw"]
        return self._prefactor * np.logaddexp(
            kde.getLogPDF() - kde.getLogMeanDensity(), self._logEpsilon
        )

    def getFreeEnergy(self):
        """
        Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number
        of collective variables. The values are in kJ/mole. The i'th position along an
        axis corresponds to minValue + i*(maxValue-minValue)/gridWidth.
        """
        return -self._kbt * self._kde["total.rw"].getLogPDF()

    def getAverageDensity(self):
        """
        Get the average density of the system as a function of the collective variables.
        """
        kde = self._kde["total" if self.exploreMode else "total.rw"]
        return np.exp(kde.getLogMeanDensity())

    def getCollectiveVariables(self, simulation):
        """
        Get the current values of all collective variables in a Simulation.

        Parameters
        ----------
        simulation
            The Simulation to query.
        """
        return self._force.getCollectiveVariableValues(simulation.context)

    def updateContext(self, context):
        """Update the collective variables in the context."""
        bias = self.getBias().value_in_unit(unit.kilojoules_per_mole)
        self._force.getTabulatedFunction(0).setFunctionParameters(
            *self._widths, bias.ravel(), *self._limits
        )
        self._force.updateParametersInContext(context)

    def addKernel(self, values, biasEnergy, variance=None):
        """Add a kernel to the PDF estimate and update the bias potential."""
        if not isinstance(biasEnergy, unit.Quantity):
            biasEnergy = biasEnergy * unit.kilojoules_per_mole
        if variance is None:
            variance = self._variance["total"].get()
        for case in self._cases:
            self._kde[case].update(values, 0.0, variance, self._lastVisitedState)
            self._kde[f"{case}.rw"].update(
                values,
                biasEnergy / self._kbt,
                variance / self._biasFactor,
                self._lastVisitedState,
            )

    def step(self, simulation, steps):
        """
        Advance the simulation by integrating a specified number of time steps.

        Parameters
        ----------
        simulation:
            The Simulation to advance.
        steps
            The number of time steps to integrate.
        """
        stepsToGo = steps
        groups = {self._force.getForceGroup()}
        while stepsToGo > 0:
            nextSteps = stepsToGo
            nextSteps = min(
                nextSteps, self._interval - simulation.currentStep % self._interval
            )
            simulation.step(nextSteps)
            if simulation.currentStep % self._interval == 0:
                position = self.getCollectiveVariables(simulation)
                for index, stateIDFunc in enumerate(self.stateIDFuncs):
                    if stateIDFunc(position):
                        self._lastVisitedState = index
                        break
                if self._adaptiveVariance:
                    self._updateSampleStats(position)
                if simulation.currentStep % self.frequency == 0:
                    state = simulation.context.getState(getEnergy=True, groups=groups)
                    energy = state.getPotentialEnergy()
                    self.addKernel(position, energy)
                    self.updateContext(simulation.context)
                    if (
                        self.saveFrequency is not None
                        and simulation.currentStep % self.saveFrequency == 0
                    ):
                        self._syncWithDisk()
            stepsToGo -= nextSteps

    def getVariance(self):
        """Get the variance of the probability distribution estimate."""
        return self._variance["total"].get()

    def getLastVisitedState(self):
        """Get the index of the last visited state."""
        return self._lastVisitedState

    def getRecollectorVariable(self, label):
        """Get the recollector function."""
        force = copy(self._force)
        kde = self._kde["total"]
        force.getTabulatedFunction(0).setFunctionParameters(
            *self._widths,
            np.exp(kde.getLogPDF(label) - kde.getLogPDF()).ravel(),
            *self._limits,
        )
        return force
        
#class OPESBiasReporter(StateDataReporter):
#    def __init__(self, file, report_interval, **kwargs):
#        self._opes_sampler = kwargs.pop('opesSampler', None)
#        super().__init__(file, report_interval, **kwargs)
#    def _initializeConstants(self, simulation):
#        super()._initializeConstants(simulation)
#        if self._opes_sampler:
#            self._opes_force = self._opes_sampler._force
#            self._opes_group = self._opes_force.getForceGroup()
#    def _constructHeaders(self):
#        headers = super()._constructHeaders()
#        if self._opes_sampler:
#            headers.append("U_bias (kJ/mol)")
#        return headers
#    def _constructReportValues(self, simulation, state):
#        values = super()._constructReportValues(simulation, state)
#        if self._opes_sampler:
#            ext_context = simulation.context.getExtensionContext()
#            bias_state = ext_context.getState(getEnergy=True, groups={self._opes_group})
#            values.append(bias_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
#        return values


class OPESBiasReporter(StateDataReporter):
    def __init__(self, file, reportInterval, opesSamplers=None, **kwargs):
        if opesSamplers is None:
            self._opesSamplers = []
        else:
            if not isinstance(opesSamplers, (list, tuple)):
                opesSamplers = [opesSamplers]

            if len(opesSamplers) > 0 and isinstance(opesSamplers[0], (list, tuple)) and len(opesSamplers[0]) == 2:
                self._opesSamplers = list(opesSamplers) 
            else:
                self._opesSamplers = [(f"opes{i}", s) for i, s in enumerate(opesSamplers)]

        self._opesForceGroups = [(name, sampler._force.getForceGroup()) for name, sampler in self._opesSamplers]

        super().__init__(file, reportInterval, **kwargs)

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        for name, _ in self._opesForceGroups:
            headers.append(f"opes_{name}_bias (kJ/mol)")
        if len(self._opesForceGroups) > 1:
            headers.append("opes_total_bias (kJ/mol)")
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)

        if len(self._opesForceGroups) > 0:
            ext_context = simulation.context.getExtensionContext()

            total = 0.0
            for _, group in self._opesForceGroups:
                st = ext_context.getState(getEnergy=True, groups={group})
                bias_kj = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                values.append(bias_kj)
                total += bias_kj

            if len(self._opesForceGroups) > 1:
                values.append(total)

        return values


