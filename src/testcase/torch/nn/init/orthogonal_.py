import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.init.orthogonal_)
class TorchNnInitOrthogonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_orthogonal__correctness(self):
        # Randomly generate the dimension of the tensor, at least 2
        dim = random.randint(2, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input_size based on the dim and num_of_elements_each_dim
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the input_size
        tensor = torch.randn(input_size)
        # Apply the orthogonal_ function
        result = torch.nn.init.orthogonal_(tensor)
        # Return the result
        return result
