
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bartlett_window)
class TorchBartlettWindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bartlett_window_correctness(self):
        window_length = random.randint(1, 10)
        periodic = random.choice([True, False])
        result = torch.bartlett_window(window_length, periodic=periodic)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bartlett_window_large_scale(self):
        window_length = random.randint(1000, 10000)
        periodic = random.choice([True, False])
        result = torch.bartlett_window(window_length, periodic=periodic)
        return result

