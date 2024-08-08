import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.outer)
class TorchOuterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_outer_correctness(self):
        # Randomly generate the size of the input vectors
        dim1 = random.randint(1, 5)
        dim2 = random.randint(1, 5)

        # Generate random input vectors
        input_vector1 = torch.randn(dim1)
        input_vector2 = torch.randn(dim2)

        # Calculate the outer product
        result = torch.outer(input_vector1, input_vector2)
        return result
