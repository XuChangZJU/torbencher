import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.index_fill_)
class TorchTensorIndexfillTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_fill__correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension to index
        dim_to_index = random.randint(0, len(input_size) - 1)
        # Generate random indices to fill, making sure the number of indices is less than the size of the dimension being indexed
        num_indices = random.randint(1, input_size[dim_to_index])
        # Generate random indices within the valid range for the chosen dimension
        indices_to_fill = torch.randint(0, input_size[dim_to_index], (num_indices,))
        # Generate a random value to fill
        value_to_fill = random.uniform(0.1, 10.0)
        # Apply index_fill_
        result = input_tensor.index_fill_(dim_to_index, indices_to_fill, value_to_fill)
        return result
    
    
    
    