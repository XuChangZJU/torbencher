import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.lu)
class TorchLinalgLuTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_lu_correctness(self):
        # Random dimension for the input matrix
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor
        A = torch.randn(input_size)
        # Calculate LU decomposition
        P, L, U = torch.linalg.lu(A)
        # Return the result of P @ L @ U
        return P @ L @ U
    
    
    # Automatically added function calls
    
    
    