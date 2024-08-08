import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.MultiLabelSoftMarginLoss)
class TorchNnMultilabelsoftmarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_multilabelsoftmarginloss_correctness(self):
        # Random batch size
        batch_size = random.randint(1, 5)
        # Random number of classes
        num_classes = random.randint(1, 5)
        # Random input tensor of shape (batch_size, num_classes)
        input_tensor = torch.randn(batch_size, num_classes)
        # Random target tensor of shape (batch_size, num_classes) with values 0 or 1
        target_tensor = torch.randint(0, 2, (batch_size, num_classes)).float()

        # Initialize the loss function
        criterion = torch.nn.MultiLabelSoftMarginLoss()
        # Compute the loss
        loss = criterion(input_tensor, target_tensor)
        return loss
