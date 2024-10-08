import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.LazyLinear)
class TorchNnLazylinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    # @unittest.skip
    def test_lazylinear_correctness(self):
        # Randomly generate input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Define input tensor
        input_tensor = torch.randn(input_size)

        # Define out_features for LazyLinear
        out_features = random.randint(1, 10)

        # Define LazyLinear module
        lazy_linear = torch.nn.LazyLinear(out_features)

        # Forward pass to initialize weights and bias
        result = lazy_linear(input_tensor)
        with torch.no_grad():
            lazy_linear.weight = torch.nn.Parameter(
                torch.randn(lazy_linear.weight.shape) * 0.01
            )
            if lazy_linear.bias is not None:
                lazy_linear.bias = torch.nn.Parameter(
                    torch.randn(lazy_linear.bias.shape) * 0.01
                )

        result = lazy_linear(input_tensor)
        return result
