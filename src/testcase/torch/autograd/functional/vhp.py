import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.functional.vhp)
class TorchAutogradFunctionalVhpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vhp_correctness(self):
        # Define a simple scalar function
        def scalar_function(x):
            return x.pow(3).sum()
    
        # Randomly generate the dimension and size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random input tensor
        inputs = torch.randn(input_size)
    
        # Generate random vector v with the same size as inputs
        v = torch.randn(input_size)
    
        # Compute the vector Hessian product
        func_output, vhp_result = torch.autograd.functional.vhp(scalar_function, inputs, v)
    
        return func_output, vhp_result
    