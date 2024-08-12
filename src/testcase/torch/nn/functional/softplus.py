import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.softplus)
class TorchNnFunctionalSoftplusTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_softplus_correctness(self):
        # Randomly generate input tensor dimension
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Generate random beta
        beta = random.uniform(0.1, 10.0)
        # Calculate softplus result
        result = torch.nn.functional.softplus(input_tensor, beta)
        return result
