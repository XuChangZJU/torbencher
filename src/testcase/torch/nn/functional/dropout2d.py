
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.dropout2d)
class Dropout2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout2d_correctness(self):
        input_data = torch.randn(10, 10, 10, 10)
        p = random.uniform(0.0, 1.0)
        training = random.choice([True, False])
        result = torch.nn.functional.dropout2d(input_data, p, training)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_dropout2d_large_scale(self):
        input_data = torch.randn(100, 100, 100, 100)
        p = random.uniform(0.0, 1.0)
        training = random.choice([True, False])
        result = torch.nn.functional.dropout2d(input_data, p, training)
        return result

