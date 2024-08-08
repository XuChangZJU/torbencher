import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.vsplit)
class TorchVsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_vsplit_correctness(self):
        # Random dimensions
        dim1 = random.randint(2, 5)
        dim2 = random.randint(2, 5)

        # Create a random tensor with at least 2 dimensions
        tensor = torch.randn(dim1, dim2)

        # Randomly choose to use integer or list for indices_or_sections
        if random.choice([True, False]):
            # Case 1: indices_or_sections is an integer that must evenly divide the number of rows
            num_sections = dim1
            while dim1 % num_sections != 0:
                num_sections = random.randint(1, dim1)
            indices_or_sections = num_sections
        else:
            # Case 2: indices_or_sections is a list or tuple of integers
            num_splits = random.randint(1, dim1 - 1)
            indices_or_sections = sorted(random.sample(range(1, dim1), num_splits))

        result = torch.vsplit(tensor, indices_or_sections)
        return result
