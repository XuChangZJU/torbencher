
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.multinomial)
class TorchMultinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multinomial(self, input=None):
        if input is not None:
            result = torch.multinomial(input[0], num_samples=input[1], replacement=input[2])
            return result
        a = torch.tensor([1., 1., 1., 1.])
        result = torch.multinomial(a, num_samples = 3, replacement = True)
        return result
        
