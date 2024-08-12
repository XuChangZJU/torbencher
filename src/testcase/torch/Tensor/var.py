import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.var)
class TorchTensorVarTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_var_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated tensor
        tensor = torch.randn(input_size)
        # Randomly generated dim
        dim = random.randint(0, len(input_size) - 1)
        # Calculate var
        result = tensor.var(dim)
        return result
