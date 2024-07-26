import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.lr_scheduler.LambdaLR)
class TorchOptimLrschedulerLambdalrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lambda_lr_correctness(self):
        # Randomly generate the number of parameter groups
        num_param_groups = random.randint(1, 5)

        # Create a list of random tensors to simulate parameters in optimizer
        params = [torch.randn(random.randint(1, 5), requires_grad=True) for _ in range(num_param_groups)]

        # Create a simple SGD optimizer with the random parameters
        optimizer = torch.optim.SGD(params, lr=random.uniform(0.01, 0.1))

        # Generate random lambda functions for each parameter group
        lr_lambdas = lambda epoch: random.uniform(0.8, 1.2) ** epoch

        # Create the LambdaLR scheduler with the optimizer and lambda functions
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)

        # Simulate a few epochs to see the effect of the scheduler
        for epoch in range(random.randint(1, 10)):
            optimizer.step()  # Simulate an optimizer step
            scheduler.step()  # Update the learning rate

        # Return the last learning rates for verification
        return scheduler.get_last_lr()
