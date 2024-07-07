import torch
import torch.fx
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fx.wrap)
class TorchFxWrapTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fx_wrap_correctness(self):
        # Define a custom function to be wrapped
        def my_custom_function(x, y):
            return x * x + y * y
    
        # Generate random dimensions and sizes for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Define a function to be traced
        def fn_to_be_traced(x, y):
            return my_custom_function(x, y)
    
        # Create a symbolic tracer
        tracer = torch.fx.Tracer()
        graph = tracer.trace(fn_to_be_traced)
    
        # Check if the custom function is preserved as a CallFunction node
        for node in graph.nodes:
            if node.op == 'call_function' and node.target == my_custom_function:
                return True
        return False
    
    
    
    