import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.flipud)
class TorchFlipudTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_flipud_4d(self, input=None):
        if input is not None:
            result = torch.flipud(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.flipud(a)
        return [result, [a]]

