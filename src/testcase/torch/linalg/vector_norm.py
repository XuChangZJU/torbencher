import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.vector_norm)
class TorchLinalgVectornormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_vector_norm_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        x = torch.randn(input_size)
        # Generate a random order for the norm
        ord = random.choice([2, float('inf'), float('-inf'), 0, random.randint(1, 10), random.uniform(1, 10)])
        # Generate a random dimension to compute the norm over
        dim_options = [None] + list(range(dim)) + [tuple(random.sample(range(dim), random.randint(1, dim)))]
        dim = random.choice(dim_options)
        # Generate a random boolean for keepdim
        keepdim = random.choice([True, False])
        # Calculate the vector norm
        result = torch.linalg.vector_norm(x, ord, dim, keepdim)
        return result
    