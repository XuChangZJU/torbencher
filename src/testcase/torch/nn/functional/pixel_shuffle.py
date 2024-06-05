
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pixel_shuffle)
class PixelShuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pixel_shuffle_correctness(self):
        input_data = torch.randn(10, 9, 10, 10)
        upscale_factor = random.randint(1, 5)
        result = torch.nn.functional.pixel_shuffle(input_data, upscale_factor)
        return result

    @test_api_version.larger_than("1.1