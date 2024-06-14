import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.StepLR)
class TorchOptimLrschedulerSteplrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_StepLR_correctness(self):
        # Define optimizer parameters
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weight = torch.randn(input_size)
        lr = random.uniform(0.01, 0.1)
        # Define optimizer
        optimizer = torch.optim.SGD([{'params': [weight], 'lr': lr}])
        # Define scheduler parameters
        step_size = random.randint(1, 10)
        gamma = random.uniform(0.1, 0.9)
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        # Run scheduler for a few epochs
        epochs = random.randint(1, 10)
        for epoch in range(epochs):
            optimizer.step()
            scheduler.step()
        # Return the last learning rate
        return optimizer.param_groups[0]['lr']
    