
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.ctc_loss)
class CTCLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss_correctness(self):
        log_probs = torch.randn(50, 10, 20)
        targets = torch.randint(0, 10, (50, 10))
        input_lengths = torch.randint(10, 20, (50,))
        target_lengths = torch.randint(5, 10, (50,))
        blank = random.randint(0, 9)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss_large_scale(self):
        log_probs = torch.randn(500, 100, 200)
        targets = torch.randint(0, 100, (500, 100))
        input_lengths = torch.randint(100, 200, (500,))
        target_lengths = torch.randint(50, 100, (500,))
        blank = random.randint(0, 99)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction)
        return result

