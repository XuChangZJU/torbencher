
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.are_deterministic_algorithms_enabled)
class TorchAreDeterministicAlgorithmsEnabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_are_deterministic_algorithms_enabled_correctness(self):
        result = torch.are_deterministic_algorithms_enabled()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_are_deterministic_algorithms_enabled_large_scale(self):
        result = torch.are_deterministic_algorithms_enabled()
        return result

