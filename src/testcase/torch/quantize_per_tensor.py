import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.quantize_per_tensor)
class TorchQuantizeUperUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_quantize_per_tensor_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        float_tensor = torch.randn(input_size)  # Random float tensor
        scale = random.uniform(0.1, 1.0)  # Random scale value between 0.1 and 1.0
        zero_point = random.randint(-128, 127)  # Random zero point value between -128 and 127
        dtype = torch.quint8  # Desired data type of returned tensor
