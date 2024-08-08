import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ReplicationPad1d)
class TorchNnReplicationpad1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ReplicationPad1d_correctness(self):
        # Randomly generate input tensor size
        batch_size = random.randint(1, 3)
        channels = random.randint(1, 3)
        width = random.randint(1, 10)
        input_size = [batch_size, channels, width]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Randomly generate padding size
        padding_left = random.randint(1, 5)
        padding_right = random.randint(1, 5)
        padding = (padding_left, padding_right)

        # Apply ReplicationPad1d
        replication_pad_1d = torch.nn.ReplicationPad1d(padding)
        result = replication_pad_1d(input_tensor)
        return result
