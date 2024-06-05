
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_complex)
class TorchIsComplexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_complex_correctness(self):
        tensor = torch.randn(random.randint(1, 10), dtype=torch.complex64)
        result = torch.is_complex(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_complex_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000), dtype=torch.complex128)
        result = torch.is_complex(tensor)
        return result

