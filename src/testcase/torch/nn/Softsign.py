
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softsign)
class TorchSoftsignTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softsign_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        softsign = torch.nn.Softsign()
        result = softsign(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_softsign_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        softsign = torch.nn.Softsign()
        result = softsign(input_tensor)
        return result

