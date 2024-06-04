
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.ndtri)
class TorchSpecialNdtriTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ndtri_0d(self):
        
        a = torch.rand([])
        result = torch.special.ndtri(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_ndtri_1d(self):
        
        a = torch.rand(5)
        result = torch.special.ndtri(a)
        return result
