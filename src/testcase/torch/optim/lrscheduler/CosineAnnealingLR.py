import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.CosineAnnealingLR)
class TorchOptimLrschedulerCosineannealinglrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_CosineAnnealingLR_correctness(self):
        # Define optimizer
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weights = torch.randn(input_size, requires_grad=True)
        optimizer = torch.optim.SGD([weights], lr=random.uniform(0.01, 0.1)) # lr is randomly generated
    
        # Define scheduler
        T_max = random.randint(2, 10) # T_max is randomly generated
        eta_min = random.uniform(0.0, 0.1) # eta_min is randomly generated
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
    
        # Get learning rate before stepping
        lr_before_step = optimizer.param_groups[0]['lr']
    
        # Step the scheduler
        scheduler.step()
    
        # Get learning rate after stepping
        lr_after_step = optimizer.param_groups[0]['lr']
    
        return lr_before_step, lr_after_step # return lr before and after stepping to show the effect of scheduler
    