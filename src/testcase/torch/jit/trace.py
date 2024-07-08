import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.jit.trace)
class TorchJitTraceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_jit_trace_correctness(self):
        # Define a simple function to trace
        def foo(x, y):
            return 2 * x + y
    
        # Generate random input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size)
        y = torch.randn(input_size)
    
        # Trace the function
        traced_foo = torch.jit.trace(foo, (x, y))
    
        # Execute the traced function
        traced_result = traced_foo(x, y)
    
        # Return the result
        return traced_result
    