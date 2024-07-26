import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ChannelShuffle)
class TorchNnChannelshuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ChannelShuffle_correctness(self):
        # Randomly generate input tensor size
        batch_size = random.randint(1, 3)
        num_channels = random.randint(2, 8)  # num_channels should be at least 2 to allow for shuffling
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        input_size = [batch_size, num_channels, height, width]

        # Randomly generate number of groups, ensuring it's a divisor of num_channels
        groups = random.choice([i for i in range(1, num_channels + 1) if num_channels % i == 0])

        input_tensor = torch.randn(input_size)
        channel_shuffle = torch.nn.ChannelShuffle(groups)
        result = channel_shuffle(input_tensor)
        return result
