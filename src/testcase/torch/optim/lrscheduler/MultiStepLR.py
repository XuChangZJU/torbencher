import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.MultiStepLR)
class TorchOptimLrschedulerMultisteplrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_MultiStepLR_correctness(self):
    # Define optimizer parameters
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    weight = torch.randn(input_size)
    lr = random.uniform(0.01, 0.1)
    # Define optimizer
    optimizer = torch.optim.SGD([weight], lr=lr)
    # Define milestones and gamma
    num_milestones = random.randint(1, 3)
    milestones = sorted(random.sample(range(1, 100), num_milestones))
    gamma = random.uniform(0.1, 0.9)
    # Define scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    # Run scheduler for some epochs
    for epoch in range(100):
        scheduler.step()
    # Get last learning rate
    last_lr = scheduler.get_last_lr()[0]
    return last_lr
