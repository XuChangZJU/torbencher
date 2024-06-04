
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pad)
class TorchNNFunctionalPadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pad_common(self):
        
        a = torch.ones(5)
        b = (0, 2)
        c = 'constant'
        d = 0.0
        result = torch.nn.functional.pad(a, pad=b, mode=c, value=d)
        return result


