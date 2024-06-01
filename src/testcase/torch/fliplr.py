import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fliplr)
class TorchFliplrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fliplr_4d(self, input=None):
        if input is not None:
            result = torch.fliplr(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.fliplr(a)
        return [result, [a]]

