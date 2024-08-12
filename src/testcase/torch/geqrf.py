import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.geqrf)
class TorchGeqrfTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_geqrf_correctness(self):
        dim1 = random.randint(1, 10)
        dim2 = random.randint(1, 10)
        input_size = [dim1, dim2]
        input = torch.randn(input_size)
        a, tau = torch.geqrf(input)
        return a
