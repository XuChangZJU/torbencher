
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.normalize)
class TorchNNFunctionalNormalizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_normalize_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.normalize(input[0], p=input[1], dim=input[2], eps=input[3], out=input[4])
            return result
        a = torch.randn(20, 100)
        b = 2.0
        c = 1
        d = 1e-12
        e = None
        result = torch.nn.functional.normalize(a, p=b, dim=c, eps=d, out=e)
        return result


