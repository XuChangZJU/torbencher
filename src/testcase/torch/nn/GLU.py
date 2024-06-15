import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GLU)
class TorchNnGluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_glu_correctness(self):
        # Generate a random input tensor size
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        
        # Initialize input_size with default single element for each dimension
        input_size = [num_of_elements_each_dim] * 5  # Assuming 5D tensor for illustration, adjust as needed
        
        # Randomly decide the dimension to split; ensure we don't pick a dimension that doesn't exist
        while True:
            dim = random.randint(-len(input_size), len(input_size) - 1)
            if dim < 0:
                dim += len(input_size)  # Convert negative index to positive
            if 0 <= dim < len(input_size):
                break  # Exit loop once a valid dimension is found
                
        # Adjust input_size to double the elements in the chosen dimension
        input_size[dim] *= 2
        
        # Create a random input tensor
        input_tensor = torch.randn(input_size)
        
        # Initialize GLU layer with the specified dimension
        glu_layer = torch.nn.GLU(dim=dim)
        
        # Apply the GLU operation using the initialized layer
        glu_result = glu_layer(input_tensor)
        
        return glu_result
    
    
    
    