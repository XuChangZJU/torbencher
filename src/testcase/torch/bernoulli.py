
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bernoulli)
class TorchBernoulliTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bernoulli(self):
        a = torch.tensor([0.3, 0.5, 0.9])
        result = torch.bernoulli(a)
        return result

