
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.flatten)
class TorchFlattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_flatten(self, input=None):
        if input is not None:
            result = torch.flatten(input[0], start_dim=input[1], end_dim=input[2])
            return [result, input]
        a = torch.randn(4, 1, 28, 28)
        result = torch.flatten(a, start_dim=1)
        return [result, [a, 1, -1]]

