
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.numel)
class TorchnumelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_numel_correctness(self):
        tensor = torch.randn(random.randint(1, 10))
        result = torch.numel(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_numel_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000))
        result = torch.numel(tensor)
        return result

