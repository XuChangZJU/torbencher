import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.put_)
class TorchTensorPutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_put__correctness(self):
        # Define the dimension and size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random tensor 
        input_tensor = torch.randn(input_size)
    
        # Generate random indices to put values
        # The number of elements in indices should be less than or equal to the number of elements in input_tensor
        num_indices = random.randint(1, torch.numel(input_tensor))
        indices = torch.randint(0, torch.numel(input_tensor), (num_indices,), dtype=torch.long)
    
        # Create a random tensor of values to put with the same number of elements as indices
        values_to_put = torch.randn(num_indices)
    
        # Use put_ to modify the input_tensor in-place
        input_tensor.put_(indices, values_to_put)
    
        # Return the modified tensor
        return input_tensor
    
    
    
    