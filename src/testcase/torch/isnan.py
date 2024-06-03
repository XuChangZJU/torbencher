
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.isnan)
class TorchIsnanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isnan(self, input=None):
        if input is not None:
            result = torch.isnan(input[0])
            return result
        a = torch.tensor([1, float('nan'), 2])
        result = torch.isnan(a)
        return result

