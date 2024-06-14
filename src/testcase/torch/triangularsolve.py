import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.triangularsolve)
class TorchTriangularsolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triangular_solve_correctness(self):
    # Randomly generate matrix dimensions
    dim1 = random.randint(1, 10)
    dim2 = random.randint(1, 10)
    # Generate random batch sizes
    batch_sizes = [random.randint(1, 5) for _ in range(random.randint(0, 3))]

    # Generate input tensors
    b_size = batch_sizes + [dim1, dim2]
    A_size = batch_sizes + [dim1, dim1]
    b = torch.randn(b_size)
    A = torch.randn(A_size)

    # Make A a upper triangular matrix
    A = torch.triu(A)

    # Call triangular_solve
    result = torch.triangular_solve(b, A)
    return result.solution
