import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.tril)
class TorchTensorTrilTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tril_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the tensors, at least 2 to form a matrix
        num_of_elements_each_dim = random.randint(2,
                                                  5)  # Random number of elements each dimension, at least 2 to form a matrix
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)
        diagonal = random.randint(-num_of_elements_each_dim + 1,
                                  num_of_elements_each_dim - 1)  # Random diagonal value within valid range
        result = tensor.tril(diagonal)
        return result
