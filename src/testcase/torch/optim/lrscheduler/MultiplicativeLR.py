import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lr_scheduler.MultiplicativeLR)
class TorchOptimLrschedulerMultiplicativelrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multiplicativelr_correctness(self):
        # Define the parameters for the optimizer
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weight = torch.randn(input_size)
        bias = torch.randn(input_size)
    
        # Define the learning rate for the optimizer
        lr = random.uniform(0.01, 0.1)
    
        # Define the optimizer
        optimizer = torch.optim.SGD([{'params': [weight], 'lr': lr},
                                    {'params': [bias], 'lr': lr}])
    
        # Define the scheduler
        lmbda = lambda epoch: random.uniform(0.8, 0.95)  # Random multiplicative factor between 0.8 and 0.95
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda)
    
        # Run the scheduler for a few epochs
        epochs = random.randint(1, 10)
        for _ in range(epochs):
            optimizer.step()
            scheduler.step()
    
        # Return the updated learning rate of the first parameter group
        return optimizer.param_groups[0]['lr']
    
    
    
    