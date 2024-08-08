import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.tensor_split)
class TorchTensorTensorUsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tensor_split_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(2, 5)  # Ensure at least 2 elements to avoid empty range error
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)

        # Randomly choose between indices or sections
        if random.choice([True, False]):
            # Randomly generate indices for splitting
            if num_of_elements_each_dim > 1:
                indices = sorted(
                    random.sample(range(1, num_of_elements_each_dim), random.randint(1, num_of_elements_each_dim - 1)))
                result = torch.tensor_split(tensor, indices, dim=0)
            else:
                result = [tensor]  # If there's only one element, no split is possible
        else:
            # Randomly generate number of sections for splitting
            sections = random.randint(1, num_of_elements_each_dim)
            result = torch.tensor_split(tensor, sections, dim=0)

        return result
