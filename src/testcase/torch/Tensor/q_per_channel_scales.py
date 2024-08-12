import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.q_per_channel_scales)
class TorchTensorQUperUchannelUscalesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_q_per_channel_scales_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        q_per_channel_axis = random.randint(0, dim - 1)  # Random q_per_channel_axis within the tensor's dimensions

        # Generate random tensor data
        tensor = torch.randn(input_size)

        # Quantize the tensor
        quantized_tensor = torch.quantize_per_channel(tensor, scales=torch.rand(input_size[q_per_channel_axis]),
                                                      zero_points=torch.randint(0, 10,
                                                                                (input_size[q_per_channel_axis],)),
                                                      axis=q_per_channel_axis, dtype=torch.qint8)

        # Get the scales of the quantized tensor
        result = quantized_tensor.q_per_channel_scales()
        return result
