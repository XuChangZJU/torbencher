import random
import torch


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.reshape)
class TorchReshapeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reshape_correctness(self):
        num_dims = random.randint(2, 4)  # Random number of dimensions between 2 and 4 for reshaping purpose
        num_elements_each_dim = random.randint(2, 5)  # Random number of elements in each dimension between 2 and 5
        input_size = [num_elements_each_dim for _ in range(num_dims)]  # Generate the input size list
        
        original_tensor = torch.randn(input_size)  # Generate a random tensor with the input size
    
        # Calculate total number of elements in the original tensor
        total_elements = 1
        for dim_size in input_size:
            total_elements *= dim_size

        # Generate a valid new shape by ensuring the total number of elements remains consistent
        new_num_dims = random.randint(1, 3)
        new_shape = [random.randint(1, 5) for _ in range(new_num_dims)]
        # Ensure the new shape is compatible
        remain_elements = total_elements
        for i in range(len(new_shape) - 1):
            candidate_size = random.randint(1, remain_elements)
            while remain_elements % candidate_size != 0:
                candidate_size -= 1
            new_shape[i] = candidate_size
            remain_elements //= new_shape[i]
        new_shape[-1] = remain_elements  # Set the last dimension to balance the total number of elements
    
        # Ensure the new shape is valid
        if remain_elements != 1:
            new_shape[-1] = remain_elements
        reshaped_tensor = torch.reshape(original_tensor, tuple(new_shape))  # Perform the reshape operation
        return reshaped_tensor
    
    
    
    