import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.isneginf)
class TorchTensorIsneginfTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_isneginf_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)
        # Introduce negative infinity values randomly in the tensor
        num_neg_inf = random.randint(1, num_of_elements_each_dim)
        for _ in range(num_neg_inf):
            idx = tuple(random.randint(0, num_of_elements_each_dim - 1) for _ in range(dim))
            tensor[idx] = float('-inf')

        result = tensor.isneginf()
        return result
