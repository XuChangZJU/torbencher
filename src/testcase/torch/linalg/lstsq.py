import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.lstsq)
class TorchLinalgLstsqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_lstsq_correctness(self):
        # Define the dimensions for the input tensors
        dim = random.randint(1, 4)
        m = random.randint(1, 5)
        n = random.randint(1, 5)
        k = random.randint(1, 5)
        input_size_A = [m, n]
        input_size_B = [m, k]
    
        # Generate random tensors A and B
        A = torch.randn(input_size_A)
        B = torch.randn(input_size_B)
    
        # Calculate the least squares solution
        solution, residuals, rank, singular_values = torch.linalg.lstsq(A, B)
    
        return solution
    