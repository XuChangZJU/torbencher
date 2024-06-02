
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReplicationPad1d)
class TorchNNReplicationPad1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_replication_pad1d(self, input=None):
        if input is not None:
            result = torch.nn.ReplicationPad1d(input[0])(input[1])
            return [result, input]
        a = torch.randn(1, 2, 4)
        pad = torch.nn.ReplicationPad1d(2)
        result = pad(a)
        return [result, [2, a]]

