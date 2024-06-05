
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dsplit)
class TorchDsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dsplit_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        indices_or_sections = random.randint(1, 10)
        result = torch.dsplit(input, indices_or_sections)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_dsplit_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000), random.randint(1000, 10000))
        indices_or_sections = random.randint(1000, 10000)
        result = torch.dsplit(input, indices_or_sections)
        return result

