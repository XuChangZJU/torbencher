import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.tile)
class TorchTileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tile_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)

        # Generate random dims, ensuring each dimension is at least 1 to avoid empty tensors
        dims = tuple(random.randint(1, 3) for _ in range(random.randint(1, 4)))
        result = torch.tile(input_tensor, dims)
        return result
