import random
import torch
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.q_per_channel_zero_points)
class TorchTensorQUperUchannelUzeroUpointsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_q_per_channel_zero_points_correctness(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        X = torch.randn(input_size)
        q_per_channel_axis = random.randint(0, dim - 1)  # Randomly choosing q_per_channel_axis
        scales = torch.rand(input_size[q_per_channel_axis])  # Generating random scales
        zero_points = torch.randint(0, 127, (input_size[q_per_channel_axis],), dtype=torch.int8)  # Adjusted range to be within int8 bounds
        quantized_tensor = torch.quantize_per_channel(X, scales, zero_points, q_per_channel_axis, torch.quint8)
        result = quantized_tensor.q_per_channel_zero_points()
        return result