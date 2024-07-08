import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fft.rfftfreq)
class TorchFftRfftfreqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_fft_rfftfreq_correctness(self):
        # Define the input size
        n = random.randint(1, 10)  # Random integer for the real FFT length
        d = random.uniform(0.1, 10.0)  # Random float for the sampling length scale
    
        # Calculate the expected output
        expected_output = torch.arange((n + 1) // 2) / (d * n)
    
        # Calculate the actual output
        output = torch.fft.rfftfreq(n, d)
    
        # Return the output
        return output
    