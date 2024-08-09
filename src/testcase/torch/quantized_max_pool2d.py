import random
import unittest
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.quantized_max_pool2d)
class TorchQuantizedUmaxUpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_quantized_max_pool2d_correctness(self):
        # Random 4D tensor dimensions (N, C, H, W)
        N = random.randint(1, 4)
        C = random.randint(1, 4)
        H = random.randint(5, 10)  # Ensuring H and W are large enough for pooling
        W = random.randint(5, 10)
        input_size = [N, C, H, W]

        # Generate a random 4D tensor
        tensor = torch.rand(input_size)
        scale = random.uniform(0.1, 2.0)  # Random scale for quantization
        zero_point = random.randint(0, 255)  # Random zero point for quantization

        # Quantize the tensor
        quantized_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, torch.quint8)

        # Random kernel_size ensuring it is smaller than the height and width of tensor
        kernel_height = random.randint(1, H)
        kernel_width = random.randint(1, W)
        kernel_size = [kernel_height, kernel_width]

        # Random stride ensuring it is smaller than the kernel size
        stride_height = random.randint(1, kernel_height)
        stride_width = random.randint(1, kernel_width)
        stride = [stride_height, stride_width]

        # Random padding ensuring it is >=0 and <= kernel_size/2
        padding_height = random.randint(0, kernel_height // 2)
        padding_width = random.randint(0, kernel_width // 2)
        padding = [padding_height, padding_width]

        # Apply quantized_max_pool2d and get the result
        result = torch.quantized_max_pool2d(quantized_tensor, kernel_size, stride, padding)
        return result
