import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.arctanh)
class TorchTensorArctanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctanh_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor data
        input_tensor = torch.randn(input_size)
        # Make sure the absolute value of each element is less than 1
        input_tensor = input_tensor / (torch.max(torch.abs(input_tensor)) + 1)
        result = input_tensor.arctanh()
        return result
