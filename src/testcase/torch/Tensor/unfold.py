import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.unfold)
class TorchTensorUnfoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_unfold_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor = torch.randn(input_size)
        unfold_dim = random.randint(0, len(input_size) - 1)  # Random valid dimension
        size = random.randint(1, input_size[unfold_dim])  # Random valid size
        step = random.randint(1, input_size[unfold_dim] - size + 1)  # Random valid step
        result = tensor.unfold(unfold_dim, size, step)
        return result
