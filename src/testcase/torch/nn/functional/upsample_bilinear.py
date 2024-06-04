
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.upsample_bilinear)
class TorchNNFunctionalUpsampleBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsample_bilinear(self):
        
        a = torch.randn(2, 3, 8, 8)
        b = (16, 16)
        c = None
        result = torch.nn.functional.upsample_bilinear(a, size=b, scale_factor=c)
        return result


