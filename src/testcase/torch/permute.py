import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.permute)
class TorchPermuteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_permute_4d(self, input=None):
        if input is not None:
            result = torch.permute(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.permute(a, (1, 0))
        return [result, [a, (1, 0)]]
