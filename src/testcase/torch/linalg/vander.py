
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vander)
class TorchLinalgVanderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_vander_correctness(self):
        dim = random.randint(2, 10)
        x = torch.randn(dim)
        N = random.randint(1, dim)
        result =