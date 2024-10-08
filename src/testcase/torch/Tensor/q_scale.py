import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.q_scale)
class TorchTensorQUscaleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_q_scale_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size)
        scale = random.uniform(0.1, 10.0)  # Random scale value between 0.1 and 10.0
        zero_point = random.randint(-128, 127)  # Random zero point value between -128 and 127
        # quantize the tensor
        quantized_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, torch.qint8)
        result = quantized_tensor.q_scale()
        return result
