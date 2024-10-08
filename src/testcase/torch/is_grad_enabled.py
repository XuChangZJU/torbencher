import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.is_grad_enabled)
class TorchIsUgradUenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_grad_enabled_correctness(self):
        # No need for random parameters, we just test the function's output in different grad states
        result_with_grad = torch.is_grad_enabled()
        with torch.no_grad():
            result_without_grad = torch.is_grad_enabled()
        return result_with_grad, result_without_grad  # Returning both to show the difference
