
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.rfft2)
class TorchRfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rfft2_correctness(self):
        dim1 = random.randint(1, 10)  # Random dimension for the tensor
        dim2 = random.randint(1, 10)  # Random dimension for the tensor
        tensor = torch.randn(dim1, dim2)
        result = torch.fft.rfft2(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rfft2_large_scale(self):
        dim1 = random.randint(1000, 10000)  # Larger random dimension for the tensor
        dim2 = random.randint(1000, 10000)  # Larger random dimension for the tensor
        tensor = torch.randn(dim1, dim2)
        result = torch.fft.rfft2(tensor)
        return result


