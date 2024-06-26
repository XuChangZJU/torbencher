
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.unique)
class TorchUniqueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unique_4d(self, input=None):
        if input is not None:
            result = torch.unique(input[0])
            return [result, input]
        a = torch.tensor([1, 2, 2, 3, 4])
        result = torch.unique(a)
        return [result, [a]]

