
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cov)
class TorchCovTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cov(self):
        input = torch.rand(10, 3)
        result = torch.cov(input, correction=1)
        return result

