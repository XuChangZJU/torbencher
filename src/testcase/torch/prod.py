
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.prod)
class TorchProdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_prod_dim(self):
        
        a = torch.randn(4, 4)
        result = torch.prod(a, 1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_prod(self):
        
        a = torch.randn(4, 4)
        result = torch.prod(a)
        return result

