import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.vmap)
class TorchVmapTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vmap_correctness(self):
        # Function to be vectorized
        def simple_function(tensor1, tensor2):
            return tensor1 + tensor2
    
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements for each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensors with matching shape
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Apply vmap on the simple function
        vmap_function = torch.vmap(simple_function)
        result = vmap_function(tensor1, tensor2)
    
        return result
    
    
    
    