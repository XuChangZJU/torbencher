
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.any)
class TorchAnyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_any_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randint(0, 2, (dim,))
        result = torch.any(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_any_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randint(0, 2, (dim,))
        result = torch.any(tensor)
        return result

