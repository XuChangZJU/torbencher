import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.eigvalsh)
class TorchLinalgEigvalshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_eigvalsh_correctness(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 10)
        # Generate a random batch size
        batch_size = random.randint(1, 5)
        # Generate a random complex or real tensor
        input_tensor = torch.randn(batch_size, dim, dim, dtype=torch.complex128)
        # Make the tensor Hermitian
        input_tensor = input_tensor + torch.conj(input_tensor.transpose(-2, -1))
        # Calculate eigenvalues
        result = torch.linalg.eigvalsh(input_tensor)
        return result
