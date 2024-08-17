import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.CTCLoss)
class TorchNnCtclossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ctc_loss_correctness(self):
        # Randomly generate dimensions and sizes
        T = random.randint(10, 50)  # Input sequence length
        C = random.randint(5, 20)  # Number of classes (including blank)
        N = random.randint(1, 16)  # Batch size
        S = random.randint(5, 30)  # Target sequence length of longest target in batch (padding length)
        S_min = random.randint(1, S - 1)  # Minimum target length

        # Initialize random batch of input vectors, for size = (T, N, C)
        input_tensor = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

        # Initialize random batch of targets (0 = blank, 1:C = classes)
        # target_tensor = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        target_tensor = torch.clamp(torch.randn(N, S) * C, min=1, max=C - 1).long()
        # Input lengths and target lengths
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        # target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        target_lengths = torch.clamp(torch.randn(N) * (S - S_min) + S_min, min=S_min, max=S - 1).long()
        # Initialize CTCLoss
        ctc_loss = torch.nn.CTCLoss()

        # Compute the loss
        loss = ctc_loss(input_tensor, target_tensor, input_lengths, target_lengths)
        return loss
