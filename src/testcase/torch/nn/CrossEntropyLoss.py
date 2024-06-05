
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CrossEntropyLoss)
class TorchCrossEntropyLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_crossentropyloss_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        target = torch.randint(0, random.randint(1, 10), (random.randint(1, 10),))
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        result = cross_entropy_loss(input_tensor, target)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_crossentropyloss_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        target = torch.randint(0, random.randint(100, 1000), (random.randint(1000, 10000),))
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        result = cross_entropy_loss(input_tensor, target)
        return result

