import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.tensorsolve)
class TorchLinalgTensorsolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tensorsolve_correctness(self):
        # Randomly generate dimensions for tensor A
        dim_A1 = random.randint(2, 4)
        dim_A2 = random.randint(2, 4)
        dim_A3 = random.randint(2, 4)
        dim_A4 = random.randint(2, 4)
        dim_A5 = random.randint(2, 4)

        # Randomly generate dimensions for tensor B
        dim_B1 = dim_A1 * dim_A2
        dim_B2 = dim_A3

        # Create tensor A with shape (dim_B1, dim_B2, dim_A4, dim_A5)
        A = torch.randn(dim_B1, dim_B2, dim_A4, dim_A5)

        # Create tensor B with shape (dim_A4, dim_A5)
        B = torch.randn(dim_A4, dim_A5)

        # Solve the tensor equation
        X = torch.linalg.tensorsolve(A, B)

        # Verify the solution
        assert torch.allclose(torch.tensordot(A, X, dims=X.ndim),
                              B), "Test failed: Solution does not satisfy the equation"

        return X
