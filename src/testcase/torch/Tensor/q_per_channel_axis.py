import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.q_per_channel_axis)
class TorchTensorQUperUchannelUaxisTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_q_per_channel_axis_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        q_per_channel_axis = random.randint(0, dim - 1)  # Randomly select the axis for per-channel quantization
        scales = torch.randn(input_size[q_per_channel_axis])  # Generate random scales for quantization
        zero_points = torch.randint(0, 256, size=(
        input_size[q_per_channel_axis],))  # Generate random zero_points for quantization
        tensor = torch.randn(input_size)
        quantized_tensor = torch.quantize_per_channel(tensor, scales, zero_points, q_per_channel_axis,
                                                      torch.quint8)  # Quantize the tensor
        result = quantized_tensor.q_per_channel_axis()
        return result
