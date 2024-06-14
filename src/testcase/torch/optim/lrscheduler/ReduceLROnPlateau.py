import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.ReduceLROnPlateau)
class TorchOptimLrschedulerReducelronplateauTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reduce_lr_on_plateau_correctness(self):
        # Define dimensions for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate a random tensor
        tensor = torch.randn(input_size)
    
        # Define learning rate
        learning_rate = random.uniform(0.01, 0.1)
    
        # Define optimizer
        optimizer = torch.optim.SGD([tensor], lr=learning_rate)
    
        # Define scheduler parameters
        mode = random.choice(['min', 'max'])
        factor = random.uniform(0.1, 0.9)
        patience = random.randint(1, 3)
        threshold = random.uniform(1e-5, 1e-3)
        threshold_mode = random.choice(['rel', 'abs'])
        cooldown = random.randint(0, 2)
        min_lr = random.uniform(1e-6, 1e-4)
        eps = random.uniform(1e-9, 1e-7)
    
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps)
    
        # Simulate training process and update learning rate
        for epoch in range(10):
            # Simulate metrics
            if mode == 'min':
                metrics = random.uniform(0.1, 1.0) - epoch * 0.1
            else:
                metrics = random.uniform(0.1, 1.0) + epoch * 0.1
            
            # Update learning rate
            scheduler.step(metrics)
    
        return optimizer.param_groups[0]['lr']
    