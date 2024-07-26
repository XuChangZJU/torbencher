import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.arctanh_)
class TorchTensorArctanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctanh_inplace_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate a random tensor with values in the range (-1, 1) to ensure valid input for arctanh
        tensor = torch.randn(input_size).clamp(-0.99, 0.99)
        result = tensor.arctanh_()
        return result
