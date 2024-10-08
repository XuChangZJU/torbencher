import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_quantized)
class TorchTensorIsUquantizedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_quantized_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list representing the size of the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)
        # Check if the tensor is quantized
        result = tensor.is_quantized
        return result

    def test_is_quantized_quantized_tensor(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list representing the size of the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)
        # Quantize the tensor
        quantized_tensor = torch.quantize_per_tensor(tensor, scale=1.0, zero_point=0, dtype=torch.qint8)
        # Check if the quantized tensor is quantized
        result = quantized_tensor.is_quantized
        return result
