
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.histc)
class TorchHistcTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_histc(self):
        
        a = torch.tensor([1., 2, 1])
        result = torch.histc(a, bins=4, min=0, max=3)
        return result

