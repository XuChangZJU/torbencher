import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.slogdet)
class TorchLinalgSlogdetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_slogdet_correctness(self):
        # Generate a random square matrix A
        dim = random.randint(1, 10)  # Random dimension for the matrix
        input_size = [dim, dim] 
        A = torch.randn(input_size)
        result = torch.linalg.slogdet(A)
        return result
    