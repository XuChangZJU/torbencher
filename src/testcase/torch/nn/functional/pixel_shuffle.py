
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pixel_shuffle)
class TorchNNFunctionalPixelShuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pixel_shuffle_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.pixel_shuffle(input[0], input[1])
            return [result, input]
        a = torch.randn(1, 9, 4, 4)
        b = 3
        result = torch.nn.functional.pixel_shuffle(a, b)
        return [result, [a, b]]


