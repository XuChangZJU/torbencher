
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.masked_select)
class TorchMasked_selectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_masked_select(self):
        
        a = torch.randn(4, 4)
        mask = a > 0
        result = torch.masked_select(a, mask)
        return result

