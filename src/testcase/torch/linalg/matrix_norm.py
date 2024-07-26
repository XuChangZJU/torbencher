import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.matrix_norm)
class TorchLinalgMatrixnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_matrix_norm_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(2, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified size
        A = torch.randn(input_size)
        # Define a list of possible ord values
        ord_values = ['fro', 'nuc', float('inf'), float('-inf'), 1, -1, 2, -2]
        # Randomly choose an ord value from the list
        ord = random.choice(ord_values)
        # Calculate the matrix norm
        result = torch.linalg.matrix_norm(A, ord)
        # Return the result
        return result
