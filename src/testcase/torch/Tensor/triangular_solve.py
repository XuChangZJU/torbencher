import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.triangular_solve)
class TorchTensorTriangularsolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triangular_solve_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(2, 4)
        num_of_elements_each_dim = random.randint(2, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensor A (must be square matrix for triangular solve)
        A_size = [input_size[0], input_size[0]]
        A = torch.randn(A_size)
    
        # Generate random tensor B (must have compatible dimensions with A)
        B_size = [input_size[0], input_size[1]]
        B = torch.randn(B_size)
    
        # Randomly choose upper, transpose, and unitriangular flags
        upper = bool(random.randint(0, 1))
        transpose = bool(random.randint(0, 1))
        unitriangular = bool(random.randint(0, 1))
    
        # Perform triangular solve
        result = torch.triangular_solve(B, A, upper=upper, transpose=transpose, unitriangular=unitriangular)
        return result
    
    
    
    