import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.float_power)
class TorchTensorFloatUpowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_float_power_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)
        # Random exponent
        exponent = random.uniform(-2, 2)
        # Calculate float power
        result = input_tensor.float_power(exponent)
        return result
