
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hardtanh)
class TorchNNFunctionalHardtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardtanh_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.hardtanh(input[0], min_val=input[1], max_val=input[2], inplace=input[3])
            return result
        a = torch.randn(4)
        b = -1.0
        c = 1
        d = False
        result = torch.nn.functional.hardtanh(a, min_val=b, max_val=c, inplace=d)
        return result


