
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.UpsamplingBilinear2d)
class TorchUpsamplingBilinear2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsamplingbilinear2d_correctness(self):
        size = (random.randint(1, 10), random.randint(1, 10))
        scale_factor = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        upsampling_bilinear = torch.nn.UpsamplingBilinear2d(size=size, scale_factor=scale_factor)
        result = upsampling_bilinear(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_upsamplingbilinear2d_large_scale(self):
        size = (random.randint(100, 1000), random.randint(100, 1000))
        scale_factor = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        upsampling_bilinear = torch.nn.UpsamplingBilinear2d(size=size, scale_factor=scale_factor)
        result = upsampling_bilinear(input_tensor)
        return result

