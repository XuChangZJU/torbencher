
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cudnn_grid_sampler)
class TorchCudnnGridSamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cudnn_grid_sampler_correctness(self):
        batch = random.randint(1, 10)
        channel = random.randint(1, 10)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        input = torch.randn(batch, channel, height, width)
        grid = torch.randn(batch, height, width, 2)
        result = torch.cudnn_grid_sampler(input, grid)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cudnn_grid_sampler_large_scale(self):
        batch = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        height = random.randint(100, 1000)
        width = random.randint(100, 1000)
        input = torch.randn(batch, channel, height, width)
        grid = torch.randn(batch, height, width, 2)
        result = torch.cudnn_grid_sampler(input, grid)
        return result

