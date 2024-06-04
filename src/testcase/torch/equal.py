
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.equal)
class TorchEqualTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_equal(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 3])
        result = torch.equal(a, b)
        return result

