import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cat)
class TorchCatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cat_4d(self, input=None):
        if input is not None:
            result = torch.cat(input)
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.cat([a, b])
        return [result, [a, b]]

