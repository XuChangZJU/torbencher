
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.fold)
class TorchNNFunctionalFoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fold_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.fold(input[0], output_size=input[1], kernel_size=input[2], dilation=input[3], padding=input[4], stride=input[5])
            return [result, input]
        a = torch.randn(1, 3 * 2 * 2, 12)
        b = (3, 4)
        c = (2, 2)
        d = 1
        e = 0
        f = 1
        result = torch.nn.functional.fold(a, output_size=b, kernel_size=c, dilation=d, padding=e, stride=f)
        return [result, [a, b, c, d, e, f]]


