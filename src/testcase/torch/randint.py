
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randint)
class TorchRandIntTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randint(self):
        result = torch.randint(low=0, high=10, size=(2, 2))
        return result

