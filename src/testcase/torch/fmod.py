
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fmod)
class TorchFmodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fmod_number(self):
        a = torch.tensor([1, 2, 3, 4, 5])
        result = torch.fmod(a, 3)
        return result
    @test_api_version.larger_than("1.1.3")
    def test_fmod(self):
        a = torch.tensor([-3, -2, -1, 1, 2, 3])
        b = torch.tensor([2, 2, 2, 2, 2, 2])
        result = torch.fmod(a, b)
        return result

