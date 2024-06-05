
import torch
import random

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.rfft)
class TorchRfftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rfft_correctness(self):
        dim = random.randint(1, 10)  # Random dimension for the tensor
        tensor = torch.randn(dim)
        result = torch.fft.rfft(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rfft_large_scale(self):
        dim = random.randint(1000, 10000)  # Larger random dimension for the tensor
        tensor = torch.randn(dim)
        result = torch.fft.rfft(tensor)
        return result


