import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.dequantize)
class TorchDequantizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_dequantize_correctness_single_tensor(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        quantized_tensor = torch.quantize_per_tensor(torch.randn(input_size), scale=1.0, zero_point=0,
                                                     dtype=torch.quint8)
        result = torch.dequantize(quantized_tensor)
        return result
