
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bernoulli)
class TorchBernoulliTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bernoulli(self, input=None):
        if input is not None:
            result = torch.bernoulli(input[0])
            return [result, input]
        a = torch.tensor([0.3, 0.5, 0.9])
        result = torch.bernoulli(a)
        return [result, [a]]

