
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.multinomial)
class TorchMultinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multinomial_correctness(self):
        input = torch.rand(random.randint(1, 10), random.randint(1, 10))
        num_samples = random.randint(1, 10)
        result = torch.multinomial(input, num_samples)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_multinomial_large_scale(self):
        input = torch.rand(random.randint(1000, 10000), random.randint(1000, 10000))
        num_samples = random.randint(1000, 10000)
        result = torch.multinomial(input, num_samples)
        return result

