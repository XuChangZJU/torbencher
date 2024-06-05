
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.complex)
class TorchComplexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_complex_correctness(self):
        real = torch.randn(random.randint(1, 10))
        imag = torch.randn(random.randint(1, 10))
        result = torch.complex(real, imag)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_complex_large_scale(self):
        real = torch.randn(random.randint(1000, 10000))
        imag = torch.randn(random.randint(1000, 10000))
        result = torch.complex(real, imag)
        return result

