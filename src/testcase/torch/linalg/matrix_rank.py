import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.matrix_rank)
class TorchLinalgMatrixUrankTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_matrix_rank_correctness(self):
        # Define the dimensions of the input tensor
        dim = random.randint(1, 4)
        m = random.randint(1, 5)
        n = random.randint(1, 5)
        input_size = [m, n]
        if dim == 2:
            input_size = [random.randint(1, 5)] + input_size
        elif dim == 3:
            input_size = [random.randint(1, 5), random.randint(1, 5)] + input_size
        elif dim == 4:
            input_size = [random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)] + input_size
        # Generate a random tensor of the specified dimensions
        A = torch.randn(input_size)
        # Calculate the matrix rank
        result = torch.linalg.matrix_rank(A)
        return result
