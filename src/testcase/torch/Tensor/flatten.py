import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.flatten)
class TorchTensorFlattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_flatten_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        start_dim = random.randint(0, dim - 1)  # Random start dimension
        end_dim = random.randint(start_dim, dim - 1)  # Random end dimension, end_dim >= start_dim
        input_tensor = torch.randn(input_size)
        result = input_tensor.flatten(start_dim, end_dim)
        return result
