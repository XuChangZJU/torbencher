import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.quantize_per_channel)
class TorchQuantizeUperUchannelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_quantize_per_channel_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        input_size = [random.randint(1, 5) for _ in range(dim)]  # Random number of elements in each dimension

        input_tensor = torch.randn(input_size).float()  # Ensure the tensor is of type Float

        # Ensure scales and zero_points dimensions align with the chosen axis
        axis = random.randint(0, dim - 1)
        scales_size = input_tensor.size(axis)

        scales = torch.tensor([random.uniform(0.01, 1.0) for _ in range(scales_size)], dtype=torch.float32)
        zero_points = torch.tensor([random.randint(0, 127) for _ in range(scales_size)], dtype=torch.int32)

        # Choose a random dtype from the allowed quantized dtypes
        dtype = random.choice([torch.quint8, torch.qint8, torch.qint32])

        result = torch.quantize_per_channel(input_tensor, scales, zero_points, axis, dtype)
        return result
