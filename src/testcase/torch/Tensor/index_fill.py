import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.index_fill)
class TorchTensorIndexUfillTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_fill_correctness(self):
        # Randomly generate tensor dimension
        dim = random.randint(0, 3)  # dim should be an integer between 0 and tensor.dim()-1
        # Randomly generate tensor size
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]

        # Generate input tensor
        input_tensor = torch.randn(input_size)
        # Generate index tensor
        index_tensor = torch.tensor([random.randint(0, input_size[dim] - 1) for i in range(random.randint(1, 5))])
        # Generate value to fill
        value = random.uniform(0.1, 10.0)

        result = input_tensor.index_fill(dim, index_tensor, value)
        return result
