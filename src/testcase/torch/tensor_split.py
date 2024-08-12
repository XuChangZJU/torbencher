import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.tensor_split)
class TorchTensorUsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tensor_split_correctness(self):
        # Generate a random dimension count for the tensor (between 1 and 4)
        dim_count = random.randint(1, 4)

        # Generate random size for each dimension (between 1 and 5)
        input_size = [random.randint(1, 5) for _ in range(dim_count)]

        # Create the tensor with the determined size
        tensor = torch.randn(input_size)

        # Randomly choose a dimension to split along (valid range is based on dim_count)
        split_dim = random.randint(0, dim_count - 1)

        # Randomly decide whether to use an integer or a list/tuple for splitting
        if random.choice([True, False]):
            # Using integer: randomly select number of sections (between 1 and size along split_dim)
            num_sections = random.randint(1, input_size[split_dim])
            result = torch.tensor_split(tensor, num_sections, split_dim)
        else:
            # Using list of indices: generate a random list of split points
            if input_size[split_dim] > 1:
                split_points = sorted(
                    random.sample(range(1, input_size[split_dim]), random.randint(1, input_size[split_dim] - 1)))
            else:
                split_points = []
            result = torch.tensor_split(tensor, split_points, split_dim)

        return result
