import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.bernoulli)
class TorchBernoulliTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_bernoulli_correctness(self):
        dim = 4  # Random dimension for the tensors
        num_of_elements_each_dim = 5  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.rand(input_size)  # probabilities
        result = torch.bernoulli(input)
        return result.shape
