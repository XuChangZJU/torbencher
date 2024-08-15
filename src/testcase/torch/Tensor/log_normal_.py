import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.log_normal_)
class TorchTensorLogUnormalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_log_normal__correctness(self):
        # Random dimension for the tensors
        dim = 4
        # Random number of elements each dimension
        num_of_elements_each_dim = 5
        # Input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Randomly generate mean and std
        mean = random.uniform(-10.0, 10.0)
        std = random.uniform(0.1, 10.0)
        # Apply log_normal_
        result = input_tensor.log_normal_(mean, std)
        return result.shape
