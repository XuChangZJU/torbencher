
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.glu)
class GLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_glu_correctness(self):
        input_data = torch.randn(10, 10)
        dim = random.randint(1, 10)
        result = torch.nn.functional.glu(input_data, dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_glu_large_scale(self):
        input_data = torch.randn(1000, 1000)
        dim = random.randint(1, 1000)
        result = torch.nn.functional.glu(input_data, dim)
        return result

