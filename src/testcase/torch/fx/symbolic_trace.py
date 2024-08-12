import random

import torch
import torch.fx

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fx.symbolic_trace)
class TorchFxSymbolicUtraceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_symbolic_trace_correctness(self):
        # Define a simple function with control flow
        def simple_function(a, b):
            if b:
                return a + 1
            else:
                return a * 2

        # Randomly choose a boolean value for concrete_args
        concrete_b = random.choice([True, False])

        # Trace the function with the concrete argument
        traced_function = torch.fx.symbolic_trace(simple_function, concrete_args={'b': concrete_b})

        # Generate a random input tensor
        input_tensor = torch.randn(random.randint(1, 5))

        # Call the traced function with the input tensor
        result = traced_function(input_tensor)

        return result
