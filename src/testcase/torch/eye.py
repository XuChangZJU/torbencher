import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.eye)
class TorchEyeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_eye_correctness(self):
        n_rows = random.randint(1, 10)  # Random number of rows between 1 and 10
        n_cols = random.randint(1, 10)  # Random number of columns between 1 and 10

        result = torch.eye(n_rows, n_cols)
        return result

    print()
