
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.ctc_loss)
class TorchCtcLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss_correctness(self):
        T = random.randint(1, 10)
        N = random.randint(1, 10)
        S = random.randint(1, 10)
        input = torch.randn(T, N, S)
        target = torch.randint(0, S, (N, random.randint(1, 10)))
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(1, target.size(1) + 1, (N,))
        blank = random.randint(0, S)
        reduction = random.choice(['none', 'mean', 'sum'])
        result = torch.ctc_loss(input, target, input_lengths, target_lengths, blank=blank, reduction=reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss_large_scale(self):
        T = random.randint(100, 1000)
        N = random.randint(100, 1000)
        S = random.randint(100, 1000)
        input = torch.randn(T, N, S)
        target = torch.randint(0, S, (N, random.randint(1, 10)))
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(1, target.size(1) + 1, (N,))
        blank = random.randint(0, S)
        reduction = random.choice(['none', 'mean', 'sum'])
        result = torch.ctc_loss(input, target, input_lengths, target_lengths, blank=blank, reduction=reduction)
        return result

