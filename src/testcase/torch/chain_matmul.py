import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.chain_matmul)
class TorchChainUmatmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chain_matmul_correctness(self):
        # Generate random dimensions for the matrices
        num_matrices = random.randint(2, 5)  # Generate at least 2 matrices
        dim1 = random.randint(1, 5)
        dim2 = random.randint(1, 5)
        dims = [dim1] + [random.randint(1, 5) for _ in range(num_matrices - 1)] + [dim2]

        # Create a list of random matrices with compatible dimensions
        matrices = [torch.randn(dims[i], dims[i + 1]) for i in range(num_matrices)]

        # Calculate the expected result using iterative matrix multiplication
        expected_result = matrices[0]
        for i in range(1, num_matrices):
            expected_result = torch.matmul(expected_result, matrices[i])

        # Calculate the result using torch.chain_matmul
        result = torch.chain_matmul(*matrices)
        return result
