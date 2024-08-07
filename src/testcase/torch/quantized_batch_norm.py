import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.quantized_batch_norm)
class TorchQuantizedUbatchUnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_quantized_batch_norm_correctness(self):
        # Random dimension sizes for the input tensor (NCHW format)
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        height = random.randint(1, 4)
        width = random.randint(1, 4)

        # Creating a random 4D input tensor and quantizing it
        input_tensor = torch.randn((batch_size, channels, height, width))
        input_scale = random.uniform(0.1, 2.0)  # Random scale for the quantized tensor
        input_zero_point = random.randint(0, 10)  # Random zero point for the quantized tensor
        quantized_input = torch.quantize_per_tensor(input_tensor, scale=input_scale, zero_point=input_zero_point,
                                                    dtype=torch.quint8)

        # Randomly generating parameters for batch normalization
        weight = torch.randn(channels)
        bias = torch.randn(channels)
        mean = torch.randn(channels)
        var = torch.randn(channels).abs()  # Variance should be positive
        eps = random.uniform(1e-5, 1e-2)  # Random epsilon value for numerical stability
        output_scale = random.uniform(0.1, 2.0)
        output_zero_point = random.randint(0, 10)

        # Applying the quantized batch normalization
        result = torch.quantized_batch_norm(quantized_input, weight, bias, mean, var, eps, output_scale,
                                            output_zero_point)
        return result
