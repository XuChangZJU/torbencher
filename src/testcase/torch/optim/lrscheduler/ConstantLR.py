import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.ConstantLR)
class TorchOptimLrschedulerConstantlrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ConstantLR_correctness(self):
    # Define optimizer parameters
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    weight = torch.randn(input_size)
    lr = random.uniform(0.01, 0.1)
    # Define optimizer
    optimizer = torch.optim.SGD([{'params': [weight], 'lr': lr}])
    # Define scheduler parameters
    factor = random.uniform(0.1, 0.9)  # Random factor between 0.1 and 0.9
    total_iters = random.randint(1, 10)  # Random total_iters between 1 and 10
    # Define scheduler
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor, total_iters)
    # Run scheduler for total_iters + 1 epochs
    for epoch in range(total_iters + 1):
        scheduler.step()
    # Get last learning rate
    last_lr = scheduler.get_last_lr()[0]
    return last_lr
