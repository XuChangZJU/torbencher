import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.TripletMarginLoss)
class TorchNnTripletmarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_triplet_margin_loss_correctness(self):
        # Randomly generate the number of samples (N) and the dimension of each sample (D)
        num_samples = random.randint(1, 10)
        dimension = random.randint(1, 128)

        # Generate random tensors for anchor, positive, and negative samples
        anchor = torch.randn(num_samples, dimension, requires_grad=True)
        positive = torch.randn(num_samples, dimension, requires_grad=True)
        negative = torch.randn(num_samples, dimension, requires_grad=True)

        # Randomly generate margin, p, and eps values
        margin = random.uniform(0.1, 10.0)
        p = random.randint(1, 3)
        eps = random.uniform(1e-7, 1e-5)

        # Create the TripletMarginLoss criterion
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=p, eps=eps)

        # Compute the loss
        result = triplet_loss(anchor, positive, negative)

        return result
