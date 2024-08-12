import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
class TorchOptimLrschedulerCosineannealingwarmrestartsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_CosineAnnealingWarmRestarts_correctness(self):
        # Define optimizer parameters
        learning_rate = random.uniform(0.001, 0.1)  # Random learning rate between 0.001 and 0.1
        params = [torch.randn(10, 5, requires_grad=True) for _ in
                  range(random.randint(1, 3))]  # Random parameters for the optimizer
        optimizer = torch.optim.SGD(params, lr=learning_rate)

        # Define scheduler parameters
        T_0 = random.randint(1, 10)  # Number of iterations for the first restart
        T_mult = random.randint(1, 3)  # Factor to increase T_i after a restart

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

        # Loop through epochs and update learning rate
        for epoch in range(random.randint(1, 20)):  # Random number of epochs between 1 and 20
            optimizer.step()
            scheduler.step()

        # Return the last learning rate
        return scheduler.get_last_lr()
