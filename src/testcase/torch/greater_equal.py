
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.greater_equal)
class TorchGreater_equalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_greater_equal(self, input=None):
        if input is not None:
            result = torch.greater_equal(input[0], input[1])
            return [result, input]
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 1, 4])
        result = torch.greater_equal(a, b)
        return [result, [a, b]]

