
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.fftfreq)
class TorchFftfreqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fftfreq_correctness(self):
        n = random.randint(1, 10)  # Random dimension for the tensor
        d = random.uniform(0.1, 10.0)  # Random alpha value between 0.1 and 10.0
        result = torch.fft.fftfreq(n, d=d)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_fftfreq_large_scale(self):
        n = random.randint(1000, 10000)  # Larger random dimension for the tensor
        d = random.uniform(0.1, 10.0)  # Random alpha value between 0.1 and 10.0
        result = torch.fft.fftfreq(n, d=d)
        return result


