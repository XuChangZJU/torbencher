
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.unsqueeze)
class TorchUnsqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unsqueeze(self):
        a = torch.tensor([1, 2, 3, 4])
        result = torch.unsqueeze(a, 1)
        return result

