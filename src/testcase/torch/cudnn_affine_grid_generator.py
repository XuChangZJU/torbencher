
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cudnn_affine_grid_generator)
class TorchCudnnAffineGridGeneratorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cudnn_affine_grid_generator_correctness(self):
        batch = random.randint(1, 10)
        channel = random.randint(1, 10)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        theta = torch.randn(batch, 2, 3)
        result = torch.cudnn_affine_grid_generator(theta, torch.Size([batch, channel, height, width]))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cudnn_affine_grid_generator_large_scale(self):
        batch = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        height = random.randint(100, 1000)
        width = random.randint(100, 1000)
        theta = torch.randn(batch, 2, 3)
        result = torch.cudnn_affine_grid_generator(theta, torch.Size([batch, channel, height, width]))
        return result

