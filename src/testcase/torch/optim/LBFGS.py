import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.LBFGS)
class TorchOptimLbfgsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LBFGS_correctness(self):
        # Define the function to optimize
        def func(x):
            return torch.norm(x)**2 
    
        # Generate random input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size, requires_grad=True)
    
        # Define the optimizer
        optimizer = torch.optim.LBFGS([x])
    
        # Define closure for LBFGS
        def closure(self):
            optimizer.zero_grad()
            output = func(x)
            output.backward()
            return output
    
        # Perform optimization
        for i in range(10):
            optimizer.step(closure)
        
        return x
        
    
    
    