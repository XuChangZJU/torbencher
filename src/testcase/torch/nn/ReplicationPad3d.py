
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReplicationPad3d)
class TorchReplicationPad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_replicationpad3d_correctness(self):
        padding = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        replication_pad = torch.nn.ReplicationPad3d(padding)
        result = replication_pad(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_replicationpad3d_large_scale(self):
        padding = (random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        replication_pad = torch.nn.ReplicationPad3d(padding)
        result = replication_pad(input_tensor)
        return result

