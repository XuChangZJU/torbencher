
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hardtanh)
class TorchNNFunctionalHardtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardtanh_common(self):
        a = torch.randn(4)
        b = -1.0
        c = 1
        d = False
        result = torch.nn.functional.hardtanh(a, min_val=b, max_val=c, inplace=d)
        return result


