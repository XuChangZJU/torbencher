import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sub)
class TorchSubTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sub_4d(self, input=None):
        if input is not None:
            result = torch.sub(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.sub(a, b, alpha=10)
        return [result, [a, b, 10]]

