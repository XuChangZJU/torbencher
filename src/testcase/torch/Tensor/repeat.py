import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.repeat)
class TorchTensorRepeatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_Tensor_repeat_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Generate random repeat sizes
        sizes = [random.randint(1, 5) for i in range(dim)]
        # Repeat the tensor
        result = input_tensor.repeat(sizes)
        return result
