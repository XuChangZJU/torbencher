
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CTCLoss)
class TorchCTCLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ctcloss_correctness(self):
        T = random.randint(1, 10)
        C = random.randint(1, 10)
        N = random.randint(1, 10)
        S_l = random.randint(1, 10)
        log_probs = torch.randn(T, N, C).log_softmax(dim=2)
        targets = torch.randint(0, C, (N, S_l))
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(1, S_l + 1, (N,))
        ctc_loss = torch.nn.CTCLoss()
        result = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_ctcloss_large_scale(self):
        T = random.randint(100, 1000)
        C = random.randint(100, 1000)
        N = random.randint(100, 1000)
        S_l = random.randint(100, 1000)
        log_probs = torch.randn(T, N, C).log_softmax(dim=2)
        targets = torch.randint(0, C, (N, S_l))
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(10, S_l + 1, (N,))
        ctc_loss = torch.nn.CTCLoss()
        result = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return result

