import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.functional.jvp)
class TorchAutogradFunctionalJvpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_jvp_correctness(self):
        # Define a random function for testing
        def random_func(x):
            return x.sin().sum(dim=0)
        
        # Generate random input tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        inputs = torch.randn(input_size)
        v = torch.randn(input_size)  # Vector for which the Jacobian vector product is computed
        
        # Compute jvp
        func_output, jvp_result = torch.autograd.functional.jvp(random_func, inputs, v)
        
        return func_output, jvp_result
    
    
    
    