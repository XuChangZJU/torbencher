
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_rng_state)
class TorchSetRngStateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_rng_state_correctness(self):
        state = torch.randn(random.randint(1, 10))
        result = torch.set_rng_state(state)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_rng_state_large_scale(self):
        state = torch.randn(random.randint(1000, 10000))
        result = torch.set_rng_state(state)
        return result

