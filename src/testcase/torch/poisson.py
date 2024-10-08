import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.poisson)
class TorchPoissonTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_poisson_correctness(self):
        dim = 4  # Random dimension for the tensors
        num_of_elements_each_dim = 5  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        rates = torch.rand(input_size) * random.uniform(0.1, 5.0)  # Rate parameter between 0 and a random value
        result = torch.poisson(rates)
        return result.shape
