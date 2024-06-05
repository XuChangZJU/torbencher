
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.binary_cross_entropy_with_logits)
class TorchBinaryCrossEntropyWithLogitsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_with_logits_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        target = torch.randint(0, 2, (dim,))
        weight = torch.randn(dim)
        reduction = random.choice(['none', 'mean', 'sum'])
        result = torch.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_with_logits_large_scale(self):
        dim = random.randint(1000, 10000)
        input = torch.randn(dim)
        target = torch.randint(0, 2, (dim,))
        weight = torch.randn(dim)
        reduction = random.choice(['none', 'mean', 'sum'])
        result = torch.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)
        return result

