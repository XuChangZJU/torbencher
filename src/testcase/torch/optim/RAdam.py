import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.RAdam)
class TorchOptimRadamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_radam_correctness(self):
    # Randomly generate the number of parameters
    num_params = random.randint(1, 5)
    
    # Randomly generate the size of each parameter tensor
    param_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
    
    # Create a list of random parameter tensors
    params = [torch.randn(param_size) for _ in range(num_params)]
    
    # Randomly generate learning rate
    lr = random.uniform(0.0001, 0.01)
    
    # Randomly generate betas
    beta1 = random.uniform(0.8, 0.99)
    beta2 = random.uniform(0.9, 0.999)
    
    # Randomly generate epsilon
    eps = random.uniform(1e-9, 1e-7)
    
    # Randomly generate weight decay
    weight_decay = random.uniform(0, 0.1)
    
    # Initialize RAdam optimizer
    optimizer = torch.optim.RAdam(params, lr, (beta1, beta2), eps, weight_decay)
    
    # Perform a single optimization step
    optimizer.step()
    
    # Return the updated parameters
    return [param.clone() for param in params]
