
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pad)
class TorchNNFunctionalPadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pad_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.pad(input[0], pad=input[1], mode=input[2], value=input[3])
            return [result, input]
        a = torch.ones(5)
        b = (0, 2)
        c = 'constant'
        d = 0.0
        result = torch.nn.functional.pad(a, pad=b, mode=c, value=d)
        return [result, [a, b, c, d]]


