
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.isnan)
class TorchIsnanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isnan(self):
        
        a = torch.tensor([1, float('nan'), 2])
        result = torch.isnan(a)
        return result

