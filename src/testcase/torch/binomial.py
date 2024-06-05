
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.binomial)
class TorchBinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_binomial_correctness(self):
        n = random.randint(1, 10)
        p = random.uniform(0.1, 10.0)
        dim = random.randint(1, 10)
        result = torch.binomial(n, p, size=(dim,))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_binomial_large_scale(self):
        n = random.randint(1, 10)
        p = random.uniform(0.1, 10.0)
        dim = random.randint(1000, 10000)
        result = torch.binomial(n, p, size=(dim,))
        return result

