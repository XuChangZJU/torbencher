
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.flatten)
class TorchFlattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_flatten(self):
        a = torch.randn(4, 1, 28, 28)
        result = torch.flatten(a, start_dim=1)
        return result

