
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.all)
class TorchAllTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_all_dim(self):
        
        a = torch.randn(4, 5, 6)
        result = torch.all(a, 1)
        return result
    @test_api_version.larger_than("1.1.3")
    def test_all(self):
        
        a = torch.ByteTensor([0, 1, 1, 0])
        result = torch.all(a)
        return result

