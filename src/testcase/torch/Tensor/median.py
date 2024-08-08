import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.median)
class TorchTensorMedianTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_Tensor_median_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random tensor
        random_tensor = torch.randn(input_size)
        # Random dim
        dim = random.randint(0, len(input_size) - 1)
        # Calculate median and indices
        median_tensor, indices = random_tensor.median(dim)
        # Return median tensor and indices
        return median_tensor, indices
