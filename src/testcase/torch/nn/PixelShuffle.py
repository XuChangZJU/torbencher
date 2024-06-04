
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PixelShuffle)
class TorchNNPixelShuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pixel_shuffle(self):
        a = torch.randn(1, 4, 8, 8)
        shuffle = torch.nn.PixelShuffle(2)
        result = shuffle(a)
        return result

