
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.hfftn)
class TorchHfftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_hfftn_5d(self):
        
        a = torch.randn(4, 3, 4, 5, 6, dtype=torch.cfloat)
        b = [2, 3, 5]
        c = [-3, -2, -1]
        e = "ortho"
        result = torch.fft.hfftn(a, b, c, e)
        return result

