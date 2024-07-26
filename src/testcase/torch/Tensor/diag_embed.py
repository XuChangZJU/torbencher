import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.diag_embed)
class TorchTensorDiagembedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diag_embed_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        tensor = torch.randn(input_size)
        # Random offset
        offset = random.randint(-max(input_size), max(input_size))  # offset should be within the range of tensor size
        # Calculate diag_embed
        result = tensor.diag_embed(offset)
        return result
