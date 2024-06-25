import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.diagonal_scatter)
class TorchTensorDiagonalscatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diagonal_scatter_correctness(self):
        # Randomly generate dimensions for the tensor
        dim1 = random.randint(0, 4)
        dim2 = random.randint(0, 4)
        while dim1 == dim2:  # Ensure dim1 and dim2 are different
            dim2 = random.randint(0, 4)
        
        # Randomly generate the size of the tensor
        tensor_size = [random.randint(2, 5) for _ in range(max(dim1, dim2) + 1)]
        
        # Create the base tensor
        base_tensor = torch.randn(tensor_size)
        
        # Create the source tensor for diagonal scatter
        min_dim = min(tensor_size[dim1], tensor_size[dim2])
        src_tensor = torch.randn(min_dim)
        
        # Randomly generate offset
        offset = random.randint(-min_dim + 1, min_dim - 1)
        
        # Perform the diagonal scatter operation
        result = base_tensor.diagonal(offset=offset, dim1=dim1, dim2=dim2).copy_(src_tensor)
        
        return result
    
    
    
    