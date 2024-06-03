
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReplicationPad3d)
class TorchNNReplicationPad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_replication_pad3d(self, input=None):
        if input is not None:
            result = torch.nn.ReplicationPad3d(input[0])(input[1])
            return result
        a = torch.randn(1, 2, 4, 4, 4)
        pad = torch.nn.ReplicationPad3d(2)
        result = pad(a)
        return result

