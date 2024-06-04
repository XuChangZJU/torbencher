
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ChannelShuffle)
class TorchNNChannelShuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_channel_shuffle(self):
        
        a = torch.randn(1, 20, 10, 10)
        cs = torch.nn.ChannelShuffle(4)
        result = cs(a)
        return result

