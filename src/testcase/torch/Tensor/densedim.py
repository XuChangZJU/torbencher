import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.densedim)
class TorchTensorDensedimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dense_dim_correctness(self):
    # Randomly decide if the tensor is sparse or not
    is_sparse = random.choice([True, False])
    
    if is_sparse:
        # Generate a random sparse tensor
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        indices = torch.randint(0, num_of_elements_each_dim, (dim, num_of_elements_each_dim))
        values = torch.randn(num_of_elements_each_dim)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=input_size)
        
        result = sparse_tensor.dense_dim()
    else:
        # Generate a random dense tensor
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        dense_tensor = torch.randn(input_size)
        
        result = dense_tensor.dense_dim()
    
    return result
