import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.softmax)
class TorchNnFunctionalSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_softmax_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random input tensor
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension to compute softmax
        dim_to_compute = random.randint(0, len(input_size) - 1)
        # Calculate softmax
        result = torch.nn.functional.softmax(input_tensor, dim_to_compute)
        return result
