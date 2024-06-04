
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.not_equal)
class TorchNotEqualTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_not_equal_number(self):
        
        a = torch.tensor([1, 2, 3])
        result = torch.not_equal(a, 2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_not_equal(self):
        
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([0, 2, 4])
        result = torch.not_equal(a, b)
        return result

