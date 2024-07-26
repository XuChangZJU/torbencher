import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.sparse_mask)
class TorchTensorSparsemaskTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sparse_mask_correctness(self):
        # Randomly generate dimensions for the tensors
        dim = random.randint(2, 4)  
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(2, 5) 
        # Create a list of dimensions
        input_size = [num_of_elements_each_dim for i in range(dim)]  
        
        # Generate a random dense tensor
        dense_tensor = torch.randn(input_size)
        
        # Generate random indices for the sparse tensor
        num_sparse_elements = random.randint(1, num_of_elements_each_dim * 2)  
        sparse_indices = torch.cat([torch.randint(0, input_size[i], size=(num_sparse_elements,)) for i in range(dim)]).reshape(dim, num_sparse_elements)
        
        # Generate random values for the sparse tensor
        sparse_values = torch.randn(num_sparse_elements, *input_size[dim:])  
        
        # Create a sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(sparse_indices, sparse_values, input_size)
        
        # Apply sparse_mask
        result = dense_tensor.sparse_mask(sparse_tensor)
        
        return result
    
    
    
    