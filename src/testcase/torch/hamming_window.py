import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.hamming_window)
class TorchHammingUwindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hamming_window_correctness(self):
        window_length = random.randint(2, 10)  # window_length must be at least 2
        periodic = random.choice([True, False])
        alpha = random.uniform(0.1, 1.0)
        beta = random.uniform(0.1, 1.0)
        result = torch.hamming_window(window_length, periodic, alpha, beta)
        return result
