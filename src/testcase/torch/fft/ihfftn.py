
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.ihfftn)
class TorchIhfftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ihfftn_correctness(self):
        dim = random.randint(1, 10)  # Random dimension for the tensor
        tensor = torch.randn(dim, dim, dim, dtype=torch.complex64)
        result = torch.fft.ihfftn(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_ihfftn_large_scale(self):
        dim = random.randint(1000, 10000)  # Larger random dimension for the tensor
        tensor = torch.randn(dim, dim, dim, dtype=torch.complex64)
        result = torch.fft.ihfftn(tensor)
        return result


