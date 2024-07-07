import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.wait)
class TorchJitWaitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_wait_correctness(self):
        # Generate random parameters for fork
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor_size = torch.Size(input_size)
        tensor1 = torch.randn(tensor_size)
        tensor2 = torch.randn(tensor_size)
    
        # Define a function to be executed asynchronously
        def async_add(a, b):
            return torch.add(a, b)
    
        # Create a Future object
        future = torch.jit.fork(async_add, tensor1, tensor2)
    
        # Wait for the Future to complete and return the result
        result = torch.jit.wait(future)
        return result
    
    
    
    