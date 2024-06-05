
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logspace)
class TorchLogspaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logspace_correctness(self):
        start = random.uniform(0.1, 10.0)
        end = random.uniform(0.1, 10.0)
        steps = random.randint(1, 10)
        base = random.uniform(0.1, 10.0)
        result = torch.logspace(start, end, steps, base=base)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_logspace_large_scale(self):
        start = random.uniform(0.1, 10.0)
        end = random.uniform(0.1, 10.0)
        steps = random.randint(1000, 10000)
        base = random.uniform(0.1, 10.0)
        result = torch.logspace(start, end, steps, base=base)
        return result

