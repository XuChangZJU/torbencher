
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.full_like)
class TorchFull_likeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_full_like(self):
        
        a = torch.randn(4)
        result = torch.full_like(a, 3.141592)
        return result

