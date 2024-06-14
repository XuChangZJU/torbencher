import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.reshape)
class TorchTensorReshapeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reshape_correctness(self):
        # Randomly generate dimensions for the original tensor
        original_dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        original_size = [num_of_elements_each_dim for _ in range(original_dim)]
        
        # Create a random tensor with the generated size
        original_tensor = torch.randn(original_size)
        
        # Calculate the total number of elements in the original tensor
        total_elements = original_tensor.numel()
        
        # Randomly generate a new shape that is compatible with the total number of elements
        new_shape = []
        remaining_elements = total_elements
        for _ in range(original_dim - 1):
            new_dim_size = random.randint(1, remaining_elements)
            new_shape.append(new_dim_size)
            remaining_elements //= new_dim_size
        new_shape.append(remaining_elements)
        
        # Reshape the tensor to the new shape
        reshaped_tensor = original_tensor.reshape(*new_shape)
        
        return reshaped_tensor
    
    
    
    