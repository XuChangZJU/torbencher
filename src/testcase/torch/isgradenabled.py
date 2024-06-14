import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.isgradenabled)
class TorchIsgradenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_grad_enabled_correctness(self):
    # No need for random parameters, we just test the function's output in different grad states
    result_with_grad = torch.is_grad_enabled()
    with torch.no_grad():
        result_without_grad = torch.is_grad_enabled()
    return result_with_grad, result_without_grad # Returning both to show the difference
