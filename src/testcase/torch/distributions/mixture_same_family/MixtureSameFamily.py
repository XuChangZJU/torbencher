import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.mixture_same_family.MixtureSameFamily)
class TorchDistributionsMixturesamefamilyMixturesamefamilyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_MixtureSameFamily_correctness(self):
        # Randomly generate batch_shape for mixture_distribution
        dim_mixture_distribution = random.randint(1, 4)
        num_of_elements_each_dim_mixture_distribution = random.randint(1, 5)
        input_size_mixture_distribution = [num_of_elements_each_dim_mixture_distribution for i in
                                           range(dim_mixture_distribution)]

        # Randomly generate num_components
        num_components = random.randint(1, 5)

        # Generate valid batch_shape for component_distribution
        input_size_component_distribution = input_size_mixture_distribution + [num_components]

        # Randomly generate parameters for Categorical distribution
        probs_mixture_distribution = torch.randn(input_size_mixture_distribution + [num_components]).abs()
        probs_mixture_distribution /= probs_mixture_distribution.sum(-1,
                                                                     keepdim=True)  # Normalize to get valid probabilities

        # Randomly generate parameters for Normal distribution (example component distribution)
        loc_component_distribution = torch.randn(input_size_component_distribution)
        scale_component_distribution = torch.rand(input_size_component_distribution) + 1e-5  # Ensure scale is positive

        # Create instances of distributions
        mixture_distribution = torch.distributions.Categorical(probs=probs_mixture_distribution)
        component_distribution = torch.distributions.Normal(loc=loc_component_distribution,
                                                            scale=scale_component_distribution)

        # Create MixtureSameFamily distribution
        mixture_same_family_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
            mixture_distribution, component_distribution)

        # Sample from the MixtureSameFamily distribution
        result = mixture_same_family_distribution.sample()
        return result
