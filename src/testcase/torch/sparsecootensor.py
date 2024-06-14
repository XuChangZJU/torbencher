import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sparsecootensor)
class TorchSparsecootensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sparse_coo_tensor_correctness(self):
        # Number of tensor dimensions
        num_dims = random.randint(2, 4)
        # Number of non-zero elements
        nnz = random.randint(1, 5)
        # Create random indices within the valid range
        indices = torch.tensor([[random.randint(0, 3) for _ in range(nnz)] for _ in range(num_dims)], dtype=torch.long)
        
        # Random values for the non-zero elements
        values = torch.randn([nnz])
        
        # Size of the sparse tensor, ensuring it can contain all indices
        size = [random.randint(3, 6) for _ in range(num_dims)]
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
    
        return sparse_tensor
    