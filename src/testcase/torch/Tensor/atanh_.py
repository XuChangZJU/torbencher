import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.atanh_)
class TorchTensorAtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atanh__correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor with values in the range (-1, 1)
        input_tensor = torch.randn(input_size) * 0.9  # Scale to ensure values are within (-1, 1)
        expected_output = torch.atanh(input_tensor)
        input_tensor.atanh_()
        return input_tensor
