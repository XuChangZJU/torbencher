
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.selu)
class TorchSeluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_selu_4d(self):
        a = torch.randn(4)
        result = torch.nn.functional.selu(a)
        return result

