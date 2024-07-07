import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.norm)
class TorchLinalgNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_norm_correctness(self):
        # Randomly choose whether to test vector or matrix norm
        is_matrix = random.choice([True, False])
        
        if is_matrix:
            # Generate random dimensions for a matrix
            dim1 = random.randint(2, 5)
            dim2 = random.randint(2, 5)
            tensor = torch.randn(dim1, dim2)
            dim = (0, 1)  # Matrix norm
        else:
            # Generate random dimensions for a vector
            dim = random.randint(1, 4)
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for _ in range(dim)]
            tensor = torch.randn(input_size)
            dim = random.randint(0, dim - 1)  # Vector norm
        
        # Randomly choose an order of norm, excluding 0 which is not supported
        ord_choices = [None, 'fro', 'nuc', float('inf'), -float('inf'), 1, -1, 2, -2, random.uniform(1.1, 3.0)]
        ord = random.choice(ord_choices)
        
        result = torch.linalg.norm(tensor, ord, dim)
        return result
    
    
    
    