import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.mode)
class TorchModeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_mode_correctness(self):
        dim = random.randint(0, 3)  # Random dimension to reduce
        keepdim = random.choice([True, False])  # Randomly choose whether to keep the reduced dimension
        input_size = [random.randint(1, 5) for _ in
                      range(dim + 1)]  # Generate random input size with at least 'dim' dimensions

        input_tensor = torch.randint(0, 10, input_size)  # Generate random tensor with values between 0 and 9
        result = torch.mode(input_tensor, dim, keepdim)
        return result
