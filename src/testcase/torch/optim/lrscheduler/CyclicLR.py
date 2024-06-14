import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.CyclicLR)
class TorchOptimLrschedulerCycliclrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_CyclicLR_correctness(self):
    # Define optimizer parameters
    learning_rate = random.uniform(0.001, 0.1)  # Random learning rate between 0.001 and 0.1
    momentum = random.uniform(0.5, 0.9)  # Random momentum between 0.5 and 0.9
    params = [torch.randn(10, 20), torch.randn(20, 30)]  # Random parameters for the optimizer

    # Define CyclicLR parameters
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
    base_lr = random.uniform(0.001, learning_rate)  # Base learning rate
    max_lr = random.uniform(learning_rate, 1.0)  # Max learning rate
    step_size_up = random.randint(100, 1000)  # Number of iterations for the increasing half of a cycle
    step_size_down = random.randint(100, 1000)  # Number of iterations for the decreasing half of a cycle
    mode = random.choice(['triangular', 'triangular2', 'exp_range'])  # Randomly choose a mode
    gamma = random.uniform(0.5, 1.5)  # Gamma value for 'exp_range' mode
    cycle_momentum = random.choice([True, False])  # Whether to cycle momentum
    base_momentum = random.uniform(0.5, momentum) if cycle_momentum else momentum  # Base momentum
    max_momentum = random.uniform(momentum, 0.9) if cycle_momentum else momentum  # Max momentum

    # Create CyclicLR scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, mode=mode, gamma=gamma, cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum)

    # Run scheduler for a few iterations
    for i in range(random.randint(1, 1000)):
        scheduler.step()

    # Return the last learning rate
    return scheduler.get_last_lr()
