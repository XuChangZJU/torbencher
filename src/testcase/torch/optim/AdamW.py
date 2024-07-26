import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.AdamW)
class TorchOptimAdamwTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adamw_correctness(self):
        # Define the parameters to be optimized
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weights = torch.randn(input_size, requires_grad=True)

        # Define a simple loss function
        def loss_fn(weights):
            return (weights ** 2).sum()

        # Create an AdamW optimizer
        lr = random.uniform(0.001, 0.1)  # Random learning rate between 0.001 and 0.1
        betas = (random.uniform(0.8, 0.999), random.uniform(0.9, 0.999))  # Random betas
        eps = random.uniform(1e-9, 1e-7)  # Random epsilon
        weight_decay = random.uniform(0.0, 0.1)  # Random weight decay
        optimizer = torch.optim.AdamW(params=[weights], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # Perform a few optimization steps
        num_steps = random.randint(1, 10)
        for _ in range(num_steps):
            loss = loss_fn(weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return weights
