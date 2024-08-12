import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.eigvals)
class TorchLinalgEigvalsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_eigvals_correctness(self):
        # Define the dimension of the square matrix
        dim = random.randint(1, 10)
        # Generate a random square matrix of complex numbers
        input_tensor = torch.randn(dim, dim, dtype=torch.complex128)
        # Calculate the eigenvalues of the matrix
        result = torch.linalg.eigvals(input_tensor)
        return result
