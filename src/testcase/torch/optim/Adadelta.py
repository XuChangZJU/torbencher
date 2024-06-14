import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Adadelta)
class TorchOptimAdadeltaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adadelta_correctness(self):
        # Randomly generate the number of parameters
        num_params = random.randint(1, 5)
        
        # Randomly generate the size of each parameter tensor
        param_sizes = [[random.randint(1, 5) for _ in range(random.randint(1, 4))] for _ in range(num_params)]
        
        # Create a list of parameter tensors
        params = [torch.randn(size) for size in param_sizes]
        
        # Randomly generate the learning rate (lr)
        lr = random.uniform(0.1, 1.0)
        
        # Randomly generate the decay rate (rho)
        rho = random.uniform(0.8, 0.99)
        
        # Randomly generate the epsilon value (eps)
        eps = random.uniform(1e-8, 1e-4)
        
        # Randomly generate the weight decay (weight_decay)
        weight_decay = random.uniform(0.0, 0.1)
        
        # Create the Adadelta optimizer
        optimizer = torch.optim.Adadelta(params, lr, rho, eps, weight_decay)
        
        # Perform a single optimization step
        optimizer.step()
        
        # Return the updated parameters
        return [param.clone() for param in params]
    