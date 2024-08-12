import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_complex)
class TorchTensorIsUcomplexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_complex_correctness(self):
        # Randomly generate dimension and number of elements for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a tensor with random float values
        float_tensor = torch.randn(input_size)
        # Check if the float tensor is complex
        result_float_tensor = float_tensor.is_complex()

        # Create a tensor with random complex values
        complex_tensor = torch.randn(input_size) + torch.randn(input_size) * 1j
        # Check if the complex tensor is complex
        result_complex_tensor = complex_tensor.is_complex()

        return result_float_tensor, result_complex_tensor
