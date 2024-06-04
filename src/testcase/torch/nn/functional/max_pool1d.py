
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.max_pool1d)
class TorchNNFunctionalMaxPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_pool1d_common(self):
        a = torch.randn(20, 16, 50)
        b = 3
        c = 2
        d = 0
        e = 1
        f = False
        g = False
        result = torch.nn.functional.max_pool1d(a, b, stride=c, padding=d, dilation=e, ceil_mode=f, return_indices=g)
        return result


