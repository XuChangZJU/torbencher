import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.diagonal)
class TorchLinalgDiagonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_diagonal_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        input_tensor = torch.randn(input_size)
        # Random offset
        offset = random.randint(-min(input_size)+1, max(input_size)-1)
        # Calculate result of torch.linalg.diagonal
        result = torch.linalg.diagonal(input_tensor, offset=offset)
        return result
    