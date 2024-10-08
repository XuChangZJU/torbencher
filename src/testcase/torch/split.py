import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.split)
class TorchSplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_split_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in
                      range(dim + 1)]  # Generate input_size, dim+1 to make sure the dimension is valid

        tensor = torch.randn(input_size)
        split_size_or_sections = random.randint(1, input_size[dim])  # Generate valid split_size_or_sections
        result = torch.split(tensor, split_size_or_sections, dim)
        return result
