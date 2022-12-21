from typing import Union

import numpy as np

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MaxValueEntropySearch, _fit_gumbel
from emukit.bayesian_optimization.interfaces import IEntropySearchModel
from emukit.core.interfaces import IModel
from emukit.core.parameter_space import ParameterSpace
from emukit.samplers import AffineInvariantEnsembleSampler
from emukit.core.initial_designs import RandomDesign


class ContinuousFidelityEntropySearch(EntropySearch):
    """
    Entropy search acquisition for continuous fidelity problems. Compared to standard entropy search,
    it computes the information gain only for the distribution of the minimum on the highest fidelity.
    """

    def __init__(
        self,
        model: Union[IModel, IEntropySearchModel],
        space: ParameterSpace,
        target_fidelity_index: int = None,
        num_samples: int = 100,
        num_representer_points: int = 50,
        burn_in_steps: int = 50,
    ):
        """
        :param model: Gaussian process model of the objective function that implements IEntropySearchModel
        :param space: Parameter space of the input domain
        :param target_fidelity_index: The index of the parameter which defines the fidelity
        :param num_samples: Integer determining how many samples to draw for each candidate input
        :param num_representer_points: Integer determining how many representer points to sample
        :param burn_in_steps: Integer that defines the number of burn-in steps when sampling the representer points
        """

        # Find fidelity parameter in parameter space
        if target_fidelity_index is None:
            self.target_fidelity_index = len(space.parameters) - 1
        else:
            self.target_fidelity_index = target_fidelity_index
        self.fidelity_parameter = space.parameters[self.target_fidelity_index]
        self.high_fidelity = self.fidelity_parameter.max

        # Sampler of representer points should sample x location at the highest fidelity
        parameters_without_info_source = space.parameters.copy()
        parameters_without_info_source.remove(self.fidelity_parameter)
        space_without_info_source = ParameterSpace(parameters_without_info_source)

        # Create sampler of representer points
        sampler = AffineInvariantEnsembleSampler(space_without_info_source)

        proposal_func = self._get_proposal_function(model, space)

        super().__init__(model, space, sampler, num_samples, num_representer_points, proposal_func, burn_in_steps)

    def _sample_representer_points(self):
        repr_points, repr_points_log = super()._sample_representer_points()

        # Add fidelity index to representer points
        idx = np.ones((repr_points.shape[0])) * self.high_fidelity
        repr_points = np.insert(repr_points, self.target_fidelity_index, idx, axis=1)
        return repr_points, repr_points_log

    def _get_proposal_function(self, model, space):

        # Define proposal function for multi-fidelity
        ei = ExpectedImprovement(model)

        def proposal_func(x):
            x_ = x[None, :]
            # Map to highest fidelity
            idx = np.ones((x_.shape[0], 1)) * self.high_fidelity

            x_ = np.insert(x_, self.target_fidelity_index, idx, axis=1)

            if space.check_points_in_domain(x_):
                val = np.log(np.clip(ei.evaluate(x_)[0], 0.0, np.PINF))
                if np.any(np.isnan(val)):
                    return np.array([np.NINF])
                else:
                    return val
            else:
                return np.array([np.NINF])

        return proposal_func


class ContinuousFidelityMaxValueEntropySearch(MaxValueEntropySearch):
    """
    Max Value Entropy search acquisition for continuous fidelity problems. Compared to standard MES,
    it computes the information gain only for the distribution of the minimum on the highest fidelity.
    """

    def __init__(
        self,
        model: Union[IModel, IEntropySearchModel],
        space: ParameterSpace,
        target_fidelity_index,
        num_samples: int = 10,
        grid_size: int = 5000,
    ):
        """
        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param target_fidelity_index: The index of the parameter which defines the fidelity
        :param num_samples: integer determining how many samples to draw of the minimum (does not need to be large)
        :param grid_size: number of random locations in grid used to fit the gumbel distribution and approximately generate
        the samples of the minimum (recommend scaling with problem dimension, i.e. 10000*d)
        """
        super().__init__(model, space, num_samples, grid_size)

        # Find fidelity parameter in parameter space
        if target_fidelity_index is None:
            self.target_fidelity_index = len(space.parameters) - 1
        else:
            self.target_fidelity_index = target_fidelity_index
        self.fidelity_parameter = space.parameters[self.target_fidelity_index]
        self.high_fidelity = self.fidelity_parameter.max

        # Sampler of representer points should sample x location at the highest fidelity
        parameters_without_info_source = space.parameters.copy()
        parameters_without_info_source.remove(self.fidelity_parameter)
        self.space_without_info_source = ParameterSpace(parameters_without_info_source)

    def update_parameters(self):
        """
        MES requires acces to a sample of possible minimum values y* of the objective function.
        To build this sample we approximate the empirical c.d.f of Pr(y*<y) with a Gumbel(a,b) distribution.
        This Gumbel distribution can then be easily sampled to yield approximate samples of y*.

        For the multi-fidelity objective, we must additionally fix s=1 when sampling points to fit our gumbel distribution.

        This needs to be called once at the start of each BO step.
        """

        # First we generate a random grid of locations at which to fit the Gumbel distribution
        random_design = RandomDesign(self.space_without_info_source)
        grid = random_design.get_samples(self.grid_size)

        # Map to highest fidelity
        grid = np.hstack([grid, np.ones((grid.shape[0], 1)) * self.high_fidelity])

        if not self.space.check_points_in_domain(grid).all():
            raise ValueError("Invalid grid points.")

        # also add the locations already queried in the previous BO steps
        grid = np.vstack([self.model.X, grid])
        # Get GP posterior at these points
        fmean, fvar = self.model.predict(grid)
        fsd = np.sqrt(fvar)

        # fit Gumbel distriubtion
        a, b = _fit_gumbel(fmean, fsd)

        # sample K times from this Gumbel distribution using the inverse probability integral transform,
        # i.e. given a sample r ~ Unif[0,1] then g = a + b * log( -1 * log(1 - r)) follows g ~ Gumbel(a,b).

        uniform_samples = np.random.rand(self.num_samples)
        gumbel_samples = np.log(-1 * np.log(1 - uniform_samples)) * b + a
        self.mins = gumbel_samples
