import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.index_put_)
class TorchTensorIndexUputUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_put_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)  # Random tensor
        indices = tuple(torch.randint(0, num_of_elements_each_dim, (num_of_elements_each_dim,)) for _ in
                        range(dim))  # Random indices
        values = torch.randn(
            num_of_elements_each_dim)  # Values tensor with the same size as the number of elements in each dimension

        result = tensor.index_put_(indices, values)  # Perform the index_put_ operation
        return result
