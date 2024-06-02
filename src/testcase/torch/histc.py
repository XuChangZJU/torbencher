
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.histc)
class TorchHistcTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_histc(self, input=None):
        if input is not None:
            result = torch.histc(input[0], bins=input[1], min=input[2], max=input[3])
            return [result, input]
        a = torch.tensor([1., 2, 1])
        result = torch.histc(a, bins=4, min=0, max=3)
        return [result, [a, 4, 0, 3]]

