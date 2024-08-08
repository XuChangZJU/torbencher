import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.acos_)
class TorchTensorAcosUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_acos__correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with values between -1 and 1
        input_tensor = torch.randn(
            input_size) * 0.9  # Multiply by 0.9 to ensure values are within the valid range for acos
        expected_result = torch.acos(input_tensor)
        input_tensor.acos_()  # In-place acos operation
        return input_tensor
