import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.slogdet)
class TorchSlogdetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_slogdet_correctness(self):
        # slogdet requires the input to be a square matrix
        dim = random.randint(1, 4)
        input_size = [dim, dim]
        input_tensor = torch.randn(input_size)
        sign, logabsdet = torch.slogdet(input_tensor)
        return sign, logabsdet
