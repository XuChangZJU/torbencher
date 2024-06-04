
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bincount)
class TorchBincountTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bincount(self):
        a = torch.randint(0, 8, (10,))
        b = torch.randn(10)
        result = torch.bincount(a, weights = b)
        return result

