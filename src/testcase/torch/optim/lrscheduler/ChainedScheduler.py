import torch
import random
from torch.optim import SGD
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, ChainedScheduler


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.ChainedScheduler)
class TorchOptimLrschedulerChainedschedulerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chained_scheduler_correctness(self):
    # Randomly generate the number of parameters and learning rate
    num_params = random.randint(1, 10)
    learning_rate = random.uniform(0.01, 0.1)
    
    # Create a random tensor for model parameters
    model_params = [torch.randn(random.randint(1, 5), requires_grad=True) for _ in range(num_params)]
    
    # Initialize the optimizer with the model parameters
    optimizer = SGD(model_params, lr=learning_rate)
    
    # Randomly generate factors and gamma for the schedulers
    factor = random.uniform(0.1, 0.5)
    total_iters = random.randint(1, 5)
    gamma = random.uniform(0.8, 0.99)
    
    # Create the individual schedulers
    scheduler1 = ConstantLR(optimizer, factor=factor, total_iters=total_iters)
    scheduler2 = ExponentialLR(optimizer, gamma=gamma)
    
    # Chain the schedulers
    chained_scheduler = ChainedScheduler([scheduler1, scheduler2])
    
    # Perform a step to update the learning rate
    chained_scheduler.step()
    
    # Return the current learning rate for the first parameter group
    return optimizer.param_groups[0]['lr']
