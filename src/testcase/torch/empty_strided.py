import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.empty_strided)
class TorchEmptyUstridedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_empty_strided_correctness(self):
        # Randomly generate valid parameters for torch.empty_strided
        dim = random.randint(1, 4)
        size = [random.randint(1, 10) for _ in range(dim)]
        stride = [random.randint(1, 10) for _ in range(dim)]  # stride should be positive integers

        # Generate the tensor using torch.empty_strided
        result = torch.empty_strided(size, stride)
        return result.shape
