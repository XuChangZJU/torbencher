
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diag)
class TorchDiagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diag_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim, dim)
        result = torch.diag(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_diag_large_scale(self):
        dim = random.randint(100, 1000)
        tensor = torch.randn(dim, dim)
        result = torch.diag(tensor)
        return result

