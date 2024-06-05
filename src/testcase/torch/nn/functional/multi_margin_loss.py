
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.multi_margin_loss)
class MultiMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multi_margin_loss_correctness(self):
        input_data = torch.randn(10, 10)
        target = torch.randint(0, 10, (10,))
        p = random.randint(1, 5)
        margin = random.uniform(0.0, 1.0)
        weight = torch.randn(10)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.multi_margin_loss(input_data, target, p, margin, weight, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_multi_margin_loss_large_scale(self):
        input_data = torch.randn(1000, 1000)
        target = torch.randint(0, 1000, (1000,))
        p = random.randint(1, 5)
        margin = random.uniform(0.0, 1.0)
        weight = torch.randn(1000)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.multi_margin_loss(input_data, target, p, margin, weight, reduction)
        return result

