import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lr_scheduler.LinearLR)
class TorchOptimLrschedulerLinearlrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LinearLR_correctness(self):
        # Define the parameters for the optimizer
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weights = torch.randn(input_size)
        optimizer = torch.optim.SGD([{'params': [weights]}], lr=random.uniform(0.01, 0.1))
    
        # Define the parameters for the scheduler
        start_factor = random.uniform(0.1, 1.0)  # start_factor should be less than or equal to 1
        end_factor = random.uniform(0.1, 1.0)  # end_factor should be less than or equal to 1
        total_iters = random.randint(1, 10)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
    
        # Run the scheduler for a few epochs and get the learning rates
        lrs = []
        for epoch in range(total_iters + 1):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
        return lrs
            
    
    
    