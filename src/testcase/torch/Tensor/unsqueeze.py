import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.unsqueeze)
class TorchTensorUnsqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_unsqueeze_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension to unsqueeze
        unsqueeze_dim = random.randint(-len(input_size) - 1, len(input_size))

        result = input_tensor.unsqueeze(unsqueeze_dim)
        return result
