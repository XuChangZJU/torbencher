
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.dropout3d)
class TorchNNFunctionalDropout3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout3d_common(self):
        
        a = torch.randn(1, 1, 1, 1, 1)
        b = 0.5
        c = True
        d = False
        result = torch.nn.functional.dropout3d(a, b, training=c, inplace=d)
        return result


