import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.hsplit)
class TorchHsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_hsplit_correctness(self):
        dim = 2  # Random tensors will be 2-dimensional for visible hsplit effect

        # Random sizes for rows and columns
        num_of_rows = random.randint(2, 5)
        num_of_cols = random.randint(4, 10)

        # Create a random 2D tensor with the generated shape
        input_tensor = torch.randn((num_of_rows, num_of_cols))

        # Randomly choose either an integer or a list of indices for `indices_or_sections`
        if random.choice([True, False]):
            while True:
                indices_or_sections = random.randint(2, num_of_cols)
                if num_of_cols % indices_or_sections == 0:
                    break
        else:
            indices_or_sections = sorted(random.sample(range(1, num_of_cols), k=random.randint(1, num_of_cols - 2)))

        result = torch.hsplit(input_tensor, indices_or_sections)
        return result
