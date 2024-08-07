import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nan_to_num)
class TorchNanUtoUnumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nan_to_num_correctness(self):
        # Define the dimension and size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor with float data type
        input_tensor = torch.randn(input_size)

        # Replace some elements with NaN, Inf, and -Inf
        input_tensor[input_tensor > 0.5] = float('nan')
        input_tensor[input_tensor < -0.5] = float('inf')
        input_tensor[input_tensor == 0] = -float('inf')

        # Generate random values for nan, posinf, and neginf
        nan_replacement = random.uniform(-10.0, 10.0)
        posinf_replacement = random.uniform(10.0, 100.0)  # Ensure greater than any other value
        neginf_replacement = random.uniform(-100.0, -10.0)  # Ensure smaller than any other value

        # Apply nan_to_num with the random replacements
        result = torch.nan_to_num(input_tensor, nan_replacement, posinf_replacement, neginf_replacement)
        return result
