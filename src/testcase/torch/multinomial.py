
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.multinomial)
class TorchMultinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multinomial(self):
        
        a = torch.tensor([1., 1., 1., 1.])
        result = torch.multinomial(a, num_samples = 3, replacement = True)
        return result
        
