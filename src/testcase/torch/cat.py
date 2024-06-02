
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cat)
class TorchCatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cat(self, input=None):
        if input is not None:
            result = torch.cat(input[0], input[1])
            return [result, input]
        a = torch.randn(2, 3)
        b = torch.randn(4, 3)
        result = torch.cat([a, b], 0)
        return [result, [[a, b], 0]]
 
