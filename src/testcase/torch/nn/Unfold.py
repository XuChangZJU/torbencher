
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Unfold)
class TorchNNUnfoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unfold(self, input=None):
        if input is not None:
            result = torch.nn.Unfold(input[0])(input[1])
            return result
        a = torch.randn(1, 3, 10, 12)
        unfold = torch.nn.Unfold(kernel_size=(3, 3))
        result = unfold(a)
        return result

