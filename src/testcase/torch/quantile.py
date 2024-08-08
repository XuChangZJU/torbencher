import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.quantile)
class TorchQuantileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_quantile_correctness(self):
        # Random dimension for the input tensor between 1 and 4 (inclusive)
        dim = random.randint(1, 4)

        # Random number of elements for the input tensor each dimension between 1 and 5 (inclusive)
        num_of_elements_each_dim = random.randint(1, 5)

        # Generating random input_size
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generating random input tensor
        input_tensor = torch.randn(input_size)

        # Random scalar or 1D tensor q for quantiles in the range [0, 1]
        q_size = random.randint(1, 4)  # Length of q tensor between 1 and 4
        q = torch.rand(q_size)  # Random 1D tensor q

        # Randomly selecting a dimension to reduce (must be a valid dimension of input_tensor)
        dim_to_reduce = random.choice(range(len(input_size)))

        # Choose a random interpolation method
        interpolation_methods = ['linear', 'lower', 'higher', 'midpoint', 'nearest']
        interpolation_method = random.choice(interpolation_methods)

        # Performing the quantile operation
        result = torch.quantile(input_tensor, q, dim_to_reduce, interpolation=interpolation_method)
        return result
