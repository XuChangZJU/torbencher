
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.multilabel_margin_loss)
class MultilabelMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multilabel_margin_loss_correctness(self):
        input_data = torch.randn(10, 10)
        target = torch.randint(0, 2, (10, 10))
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.multilabel_margin_loss(input_data, target, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_multilabel_margin_loss_large_scale(self):
        input_data = torch.randn(1000, 1000)
        target = torch.randint(0, 2, (1000, 1000))
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.multilabel_margin_loss(input_data, target, reduction)
        return result

