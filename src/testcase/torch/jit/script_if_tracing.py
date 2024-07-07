import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.script_if_tracing)
class TorchJitScriptiftracingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_script_if_tracing_correctness(self):
        # Define a simple function to be compiled
        def my_function(x, y):
            return x + y
    
        # Generate random input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Compile the function using torch.jit.script_if_tracing
        compiled_fn = torch.jit.script_if_tracing(my_function)
    
        # Execute the compiled function
        result_tracing = compiled_fn(tensor1, tensor2)
    
        # Execute the original function
        result_no_tracing = my_function(tensor1, tensor2)
    
        # Check if the results are the same
        assert torch.allclose(result_tracing, result_no_tracing)
    
        return result_tracing
        
    
    
    