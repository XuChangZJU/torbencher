
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveAvgPool2d)
class TorchAdaptiveAvgPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptiveavgpool2d_correctness(self):
        batch_size = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        input_height = random.randint(1, 10)
        input_width = random.randint(1, 10)
        output_height = random.randint(1, input_height)
        output_width = random.randint(1, input_width)
        input_tensor = torch.randn(batch_size, in_channels, input_height, input_width)
        adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((output_height, output_width))
        result = adaptive_avg_pool(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_adaptiveavgpool2d_large_scale(self):
        batch_size = random.randint(1000, 10000)
        in_channels = random.randint(100, 1000)
        input_height = random.randint(1000, 10000)
        input_width = random.randint(1000, 10000)
        output_height = random.randint(100, input_height)
        output_width = random.randint(100, input_width)
        input_tensor = torch.randn(batch_size, in_channels, input_height, input_width)
        adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((output_height, output_width))
        result = adaptive_avg_pool(input_tensor)
        return result

