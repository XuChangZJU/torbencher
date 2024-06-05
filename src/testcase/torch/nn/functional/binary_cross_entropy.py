
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.binary_cross_entropy)
class BinaryCrossEntropyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_correctness(self):
        input_data = torch.randn(10, 10)
        target = torch.randint(0, 2, (10, 10))
        weight = torch.randn(10, 10)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.binary_cross_entropy(input_data, target, weight, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_large_scale(self):
        input_data = torch.randn(1000, 1000)
        target = torch.randint(0, 2, (1000, 1000))
        weight = torch.randn(1000, 1000)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.binary_cross_entropy(input_data, target, weight, reduction)
        return result

