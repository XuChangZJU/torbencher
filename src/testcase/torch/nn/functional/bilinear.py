
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.bilinear)
class TorchNNFunctionalBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bilinear_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.bilinear(input[0], input[1], input[2])
            return result
        a = torch.randn(1, 1, 1, 2)
        b = torch.randn(1, 2, 2, 1)
        c = torch.randn(1, 3, 1, 1)
        result = torch.nn.functional.bilinear(a, b, c)
        return result


