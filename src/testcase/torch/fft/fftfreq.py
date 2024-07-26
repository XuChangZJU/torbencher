import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.fftfreq)
class TorchFftFftfreqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_fft_fftfreq_correctness(self):
        # Random integer for the FFT length
        n = random.randint(1, 10)
        # Random float for the sampling length scale
        d = random.uniform(0.1, 10.0)
        # Calculate the expected output
        expected_output = [i if i <= (n - 1) // 2 else i - n for i in range(n)]
        expected_output = torch.tensor(expected_output) / (d * n)
        # Calculate the actual output
        result = torch.fft.fftfreq(n, d)
        return result
