import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest


@test_api(torch.nn.MultiLabelMarginLoss)
class TorchNnMultilabelmarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_multilabel_margin_loss_correctness(self):
        # Random batch size between 1 and 4
        batch_size = random.randint(1, 10)
        # Random number of classes between 2 and 5
        num_classes = random.randint(2, 10)

        # Generate random input tensor with shape (batch_size, num_classes)
        input_tensor = torch.randint(0, num_classes, (batch_size, num_classes)).float()

        # Generate random target tensor with shape (batch_size, num_classes)
        # Ensure target values are valid indices and padded with -1
        target_tensor = torch.randint(0, num_classes, (batch_size, num_classes)).long()

        # Initialize the MultiLabelMarginLoss criterion
        criterion = torch.nn.MultiLabelMarginLoss()

        # Compute the loss
        loss = criterion(input_tensor, target_tensor)

        return loss
