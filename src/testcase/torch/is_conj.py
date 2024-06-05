
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_conj)
class TorchIsConjTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_conj_correctness(self):
        tensor = torch.randn(random.randint(1, 10)).conj()
        result = torch.is_conj(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_conj_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000)).conj()
        result = torch.is_conj(tensor)
        return result

