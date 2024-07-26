import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.lr_scheduler.ExponentialLR)
class TorchOptimLrschedulerExponentiallrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exponentiallr_correctness(self):
        # Define parameters for the optimizer and scheduler
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weight = torch.randn(input_size)
        lr = random.uniform(0.01, 0.1)
        gamma = random.uniform(0.1, 0.9)

        # Create an optimizer and scheduler
        optimizer = torch.optim.SGD([weight], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        # Step the scheduler for a few epochs and store learning rates
        learning_rates = []
        num_epochs = random.randint(1, 5)
        for _ in range(num_epochs):
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]['lr'])

        return learning_rates
