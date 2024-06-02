
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.gather)
class TorchGatherTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gather(self, input=None):
        if input is not None:
            result = torch.gather(input[0], input[1], input[2])
            return [result, input]
        a = torch.tensor([[1, 2], [3, 4]])
        b = 1
        c = torch.tensor([[0, 0], [1, 0]])
        result = torch.gather(a, b, c)
        return [result, [a, b, c]]

