import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.values)
class TorchTensorValuesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_values_correctness(self):
        # Randomly generate dimensions for the sparse tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Generate random indices for the sparse tensor
        num_nonzero_elements = random.randint(1, num_of_elements_each_dim)
        indices = torch.randint(0, num_of_elements_each_dim, (dim, num_nonzero_elements))
        
        # Generate random values for the sparse tensor
        values = torch.randn(num_nonzero_elements)
        
        # Create a sparse COO tensor
        sparse_tensor = torch.sparse_coo_tensor(indices, values, input_size).coalesce()
        
        # Get the values tensor from the sparse COO tensor
        result = sparse_tensor.values()
        return result
    