import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lr_scheduler.PolynomialLR)
class TorchOptimLrschedulerPolynomiallrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_PolynomialLR_correctness(self):
        # Define optimizer parameters
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weight = torch.randn(input_size)
        lr = random.uniform(0.001, 0.1)  # Random learning rate between 0.001 and 0.1
        optimizer = torch.optim.SGD([{'params': [weight], 'lr': lr}])
    
        # Define scheduler parameters
        total_iters = random.randint(1, 10)  # Random total iterations between 1 and 10
        power = random.uniform(0.5, 2.0)  # Random power between 0.5 and 2.0
    
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters, power)
    
        # Run scheduler for a few epochs and store learning rates
        lrs = []
        for _ in range(total_iters + 1):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
    
        return lrs
    
    
    
    