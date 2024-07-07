import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.ldl_factor_ex)
class TorchLinalgLdlfactorexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ldl_factor_ex_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(2, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Create a random symmetric matrix
        A = torch.randn(input_size)
        A = A @ A.mT  # Make symmetric
        
        # Perform LDL factorization
        LD, pivots, info = torch.linalg.ldl_factor_ex(A)
        
        return LD, pivots, info
    
    
    
    