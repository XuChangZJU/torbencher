
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.kl_div)
class TorchNNFunctionalKLDivTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kl_div_common(self):
        
        a = torch.randn(4, 2)
        b = torch.randn(4, 2)
        c = True
        d = True
        e = 'mean'
        result = torch.nn.functional.kl_div(a, b, size_average=c, reduce=d, reduction=e)
        return result


