import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cholesky)
class TorchTensorCholeskyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_correctness(self):
        # Generate a random positive-definite matrix
        dim = random.randint(1, 4)
        a = torch.randn(dim, dim)
        input_tensor = torch.mm(a, a.t())  # Multiplying a matrix by its transpose guarantees positive-definiteness
        result = input_tensor.cholesky()
        return result
    