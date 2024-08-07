import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.pca_lowrank)
class TorchPcaUlowrankTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pca_lowrank_correctness(self):
        dim1 = random.randint(1, 10)  # Random dimension for the tensor
        dim2 = random.randint(1, 10)  # Random dimension for the tensor
        input_size = [dim1, dim2]
        A = torch.randn(input_size)  # The input tensor
        U, S, V = torch.pca_lowrank(A)
        return U @ torch.diag(S) @ V.T  # Reconstruct the original matrix from U, S, and V
