import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version


# @test_api(torch.jit....) TODO 待查
class TorchJitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_jit_trace_correctness(self):
        # Define a simple function to be traced
        def my_function(x):
            return x * 2

        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        input_tensor = torch.randn(input_size)

        # Trace the function
        traced_function = torch.jit.trace(my_function, input_tensor)

        # Run the traced function
        result = traced_function(input_tensor)
        return result
