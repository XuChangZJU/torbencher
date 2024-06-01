import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.stack)
class TorchStackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stack_4d(self, input=None):
        if input is not None:
            result = torch.stack(input)
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.stack([a, b])
        return [result, [a, b]]

