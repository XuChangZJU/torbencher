import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.init.sparse_)
class TorchNnInitSparseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sparse_correctness(self):
        # Randomly generate dimensions for a 2D tensor
        rows = random.randint(1, 10)
        cols = random.randint(1, 10)
        tensor_size = [rows, cols]
    
        # Create a random 2D tensor
        tensor = torch.empty(tensor_size)
    
        # Randomly generate sparsity value between 0 and 1
        sparsity = random.uniform(0.0, 1.0)
    
        # Randomly generate standard deviation for the normal distribution
        std = random.uniform(0.01, 1.0)
    
        # Initialize the tensor as a sparse matrix
        result = torch.nn.init.sparse_(tensor, sparsity, std)
        return result
    
    
    
    