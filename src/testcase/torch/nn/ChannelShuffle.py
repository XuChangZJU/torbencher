
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ChannelShuffle)
class TorchChannelShuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_channelshuffle_correctness(self):
        groups = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), groups * random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        channel_shuffle = torch.nn.ChannelShuffle(groups=groups)
        result = channel_shuffle(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_channelshuffle_large_scale(self):
        groups = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), groups * random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        channel_shuffle = torch.nn.ChannelShuffle(groups=groups)
        result = channel_shuffle(input_tensor)
        return result

