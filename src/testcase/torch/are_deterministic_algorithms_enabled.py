import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.are_deterministic_algorithms_enabled)
class TorchAreUdeterministicUalgorithmsUenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_are_deterministic_algorithms_enabled_correctness(self):
        # Test with deterministic algorithms enabled
        torch.use_deterministic_algorithms(True)
        result_true = torch.are_deterministic_algorithms_enabled()

        # Test with deterministic algorithms disabled
        torch.use_deterministic_algorithms(False)
        result_false = torch.are_deterministic_algorithms_enabled()

        return result_true, result_false
