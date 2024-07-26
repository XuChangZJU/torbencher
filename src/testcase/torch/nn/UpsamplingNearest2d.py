import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.UpsamplingNearest2d)
class TorchNnUpsamplingnearest2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsamplingnearest2d_correctness(self):
        # Randomly generate input tensor size
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        input_height = random.randint(1, 10)
        input_width = random.randint(1, 10)
        input_size = [batch_size, channels, input_height, input_width]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Randomly generate scale_factor
        scale_factor_height = random.uniform(1, 3)
        scale_factor_width = random.uniform(1, 3)

        # Define UpsamplingNearest2d module
        upsampling = torch.nn.UpsamplingNearest2d(scale_factor=(scale_factor_height, scale_factor_width))

        # Perform upsampling operation
        result = upsampling(input_tensor)

        return result
