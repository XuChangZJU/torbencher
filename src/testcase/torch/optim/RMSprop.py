import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.RMSprop)
class TorchOptimRmspropTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rmsprop_correctness(self):
        # Randomly generate the size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Randomly generate the input tensor
        input_tensor = torch.randn(input_size, requires_grad=True)
    
        # Randomly generate the learning rate
        lr = random.uniform(0.001, 0.1)
    
        # Randomly generate the momentum factor
        momentum = random.uniform(0.0, 0.9)
    
        # Randomly generate the smoothing constant
        alpha = random.uniform(0.9, 0.99)
    
        # Randomly generate the term added to the denominator to improve numerical stability
        eps = random.uniform(1e-9, 1e-7)
    
        # Randomly decide whether to use centered RMSProp
        centered = random.choice([True, False])
    
        # Randomly generate the weight decay
        weight_decay = random.uniform(0.0, 0.1)
    
        # Define a simple objective function
        def objective(tensor):
            return torch.sum(tensor ** 2)
    