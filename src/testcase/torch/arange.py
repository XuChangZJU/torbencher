
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arange)
class TorchArangeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arange_one_param(self):
        result = torch.arange(5)
        return result
    @test_api_version.larger_than("1.1.3")
    def test_arange(self):
        result = torch.arange(1, 2.5, 0.5)
        return result

