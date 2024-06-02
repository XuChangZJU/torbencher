
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.broadcast_tensors)
class TorchBroadcast_tensorsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_tensors(self, input=None):
        if input is not None:
            result = torch.broadcast_tensors(input[0], input[1])
            return [result[0], input]
        x = torch.arange(3).view(1, 3)
        y = torch.arange(2).view(2, 1)
        result = torch.broadcast_tensors(x, y)
        return [result[0], [x, y]]

