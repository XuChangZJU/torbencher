
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.any)
class TorchAnyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_any_dim(self, input=None):
        if input is not None:
            result = torch.any(input[0], input[1])
            return result
        a = torch.randn(4, 5, 6)
        result = torch.any(a, 1)
        return result
    
    @test_api_version.larger_than("1.1.3")
    def test_any(self, input=None):
        if input is not None:
            result = torch.any(input[0])
            return result
        a = torch.ByteTensor([0, 1, 1, 0])
        result = torch.any(a)
        return result

