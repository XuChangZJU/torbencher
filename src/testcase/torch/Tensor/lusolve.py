import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.lusolve)
class TorchTensorLusolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_solve_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(2, 4)
    num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements each dimension

    # Generate a random tensor for LU_data
    LU_data_size = [num_of_elements_each_dim for _ in range(dim)]
    LU_data = torch.randn(LU_data_size)

    # Generate a random tensor for LU_pivots
    LU_pivots_size = [LU_data_size[0]]  # LU_pivots should match the first dimension of LU_data
    LU_pivots = torch.randint(1, LU_data_size[0] + 1, LU_pivots_size)

    # Generate a random tensor for B (right-hand side matrix)
    B_size = LU_data_size[:-1] + [random.randint(1, 5)]  # B should match the dimensions of LU_data except the last one
    B = torch.randn(B_size)

    # Perform LU solve
    result = B.lu_solve(LU_data, LU_pivots)
    return result
