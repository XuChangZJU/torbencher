import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.expand)
class TorchTensorExpandTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_expand_correctness(self):
        # Randomly generate dimensions for the original tensor
        original_dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        original_size = [num_of_elements_each_dim for _ in range(original_dim)]
    
        # Create a random tensor with the generated size
        original_tensor = torch.randn(original_size)
    
        # Generate new sizes for expansion
        expanded_size = []
        for _ in range(original_dim):
            if random.choice([True, False]):
                expanded_size.append(-1)  # Keep the original size
            else:
                expanded_size.append(random.randint(1, 10))  # Expand to a larger size
    
        # Append new dimensions at the front if necessary
        new_dims = random.randint(0, 2)
        for _ in range(new_dims):
            expanded_size.insert(0, random.randint(1, 10))
    
        # Perform the expand operation
        result = original_tensor.expand(*expanded_size)
        return result
    
    
    
    