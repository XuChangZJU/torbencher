
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PixelUnshuffle)
class TorchNNPixelUnshuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pixel_unshuffle(self, input=None):
        if input is not None:
            result = torch.nn.PixelUnshuffle(input[0])(input[1])
            return [result, input]
        a = torch.randn(1, 4, 8, 8)
        unshuffle = torch.nn.PixelUnshuffle(2)
        result = unshuffle(a)
        return [result, [2, a]]

