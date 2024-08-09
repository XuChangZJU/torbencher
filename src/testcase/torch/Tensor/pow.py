import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.pow)
class TorchTensorPowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_pow_correctness(self):
        # Generate random dimension and size for the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor and exponent
        input_tensor = torch.abs(torch.randn(input_size)) + 1e-5
        exponent = random.uniform(-2.0, 3.0)  # Random exponent between -2.0 and 3.0

        # Calculate the power of the tensor
        result = input_tensor.pow(exponent)
        return result
