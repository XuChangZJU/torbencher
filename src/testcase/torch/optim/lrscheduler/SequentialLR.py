import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.SequentialLR)
class TorchOptimLrschedulerSequentiallrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_SequentialLR_correctness(self):
    # Define optimizer parameters
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    weight = torch.randn(input_size)
    # bias should be a one-dimensional tensor
    bias = torch.randn([random.randint(1, 5)])
    optimizer = torch.optim.SGD(lr=random.uniform(0.01, 0.1), params=[{'params': weight}, {'params': bias}])

    # Define scheduler1 parameters
    factor = random.uniform(0.1, 0.9)
    total_iters = random.randint(1, 5)
    scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=factor, total_iters=total_iters)

    # Define scheduler2 parameters
    gamma = random.uniform(0.1, 0.9)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Define SequentialLR parameters
    schedulers = [scheduler1, scheduler2]
    # milestones should be a list with at least one element and the first element should be larger than total_iters
    milestones = [total_iters + random.randint(1, 5)]
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones)

    # Run scheduler for a few epochs and record learning rates
    learning_rates = []
    for epoch in range(10):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])
    
    return learning_rates
