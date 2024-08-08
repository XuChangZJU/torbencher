import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.ASGD)
class TorchOptimAsgdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_asgd_correctness(self):
        # Define the parameters for the optimizer
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weights = torch.randn(input_size, requires_grad=True)
        lr = random.uniform(0.01, 0.1)  # Learning rate
        lambd = random.uniform(1e-5, 1e-3)  # Decay term
        alpha = random.uniform(0.6, 0.9)  # Power for eta update
        t0 = random.uniform(1e5, 1e7)  # Point at which to start averaging

        # Create an ASGD optimizer
        optimizer = torch.optim.ASGD(params=[weights], lr=lr, lambd=lambd, alpha=alpha, t0=t0)

        # Define a simple loss function
        def loss_fn(weights):
            return (weights * torch.randn(input_size)).sum()

        # Perform a few optimization steps
        num_steps = random.randint(5, 10)
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(weights)
            loss.backward()
            optimizer.step()

        return weights
