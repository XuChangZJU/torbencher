
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pdist)
class PdistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pdist_correctness(self):
        input_data = torch.randn(10, 10)
        p = random.randint(1, 5)
        result = torch.nn.functional.pdist(input_data, p)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_pdist_large_scale(self):
        input_data = torch.randn(1000, 1000)
        p = random.randint(1, 5)
        result = torch.nn.functional.pdist(input_data, p)
        return result

