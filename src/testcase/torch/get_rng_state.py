
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.get_rng_state)
class TorchGetRngStateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_rng_state_correctness(self):
        result = torch.get_rng_state()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_rng_state_large_scale(self):
        result = torch.get_rng_state()
        return result

