import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.asin)
class TorchTensorAsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_asin_correctness(self):
        # Generate random dimension and size for the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with values between -1 and 1
        input_tensor = torch.rand(input_size) * 2 - 1  # Scale and shift to be in range [-1, 1]

        # Calculate arcsine using torch.Tensor.asin()
        result = input_tensor.asin()

        return result
