
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dropout)
class TorchDropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        p = random.uniform(0.1, 10.0)
        training = random.choice([True, False])
        result = torch.dropout(input, p=p, training=training)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_dropout_large_scale(self):
        dim = random.randint(1000, 10000)
        input = torch.randn(dim)
        p = random.uniform(0.1, 10.0)
        training = random.choice([True, False])
        result = torch.dropout(input, p=p, training=training)
        return result

