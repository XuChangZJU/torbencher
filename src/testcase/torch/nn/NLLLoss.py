import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.NLLLoss)
class TorchNnNlllossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    # @unittest.skip
    def test_nllloss_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 5)  # Random batch size
        num_classes = random.randint(2, 10)  # Random number of classes
        height = random.randint(1, 10)  # Random height for 2D case
        width = random.randint(1, 10)  # Random width for 2D case

        # Generate random input tensor with log-probabilities
        input_tensor = torch.randn(batch_size, num_classes, height, width, requires_grad=True)

        # Generate random target tensor with class indices
        # target_tensor = torch.empty(batch_size, height, width, dtype=torch.long).random_(0, num_classes)

        # Generate random target tensor with class indices
        target_tensor = torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.long)

        # Initialize LogSoftmax and NLLLoss modules
        log_softmax = torch.nn.LogSoftmax(dim=1)
        nll_loss = torch.nn.NLLLoss()

        # Apply LogSoftmax to input tensor
        log_probs = log_softmax(input_tensor)

        # Compute NLLLoss
        loss = nll_loss(log_probs, target_tensor)

        return loss
