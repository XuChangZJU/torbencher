import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.is_deterministic_algorithms_warn_only_enabled)
class TorchIsdeterministicalgorithmswarnonlyenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_deterministic_algorithms_warn_only_enabled_correctness(self):
        # Test the correctness of torch.is_deterministic_algorithms_warn_only_enabled
        torch.use_deterministic_algorithms(mode=True, warn_only=True)  # set warn_only to True
        result = torch.is_deterministic_algorithms_warn_only_enabled()
        return result
