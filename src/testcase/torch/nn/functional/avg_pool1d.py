
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.avg_pool1d)
class TorchNNFunctionalAvgPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool1d_common(self):
        
        a = torch.randn(1, 3, 8)
        b = 2
        c = 2
        d = 0
        e = False
        f = True
        result = torch.nn.functional.avg_pool1d(a, b, stride=c, padding=d, ceil_mode=e, count_include_pad=f)
        return result


