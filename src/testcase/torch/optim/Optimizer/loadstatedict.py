import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Optimizer.loadstatedict)
class TorchOptimOptimizerLoadstatedictTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_load_state_dict_correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create random tensors
        tensor1 = torch.randn(input_size, requires_grad=True)
        tensor2 = torch.randn(input_size, requires_grad=True)
    
        # Define a simple model
        linear = torch.nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim)
    
        # Create an optimizer
        optimizer = torch.optim.Adam([tensor1, tensor2], lr=0.1)
    
        # Perform a few optimization steps
        for i in range(5):
            loss = linear(tensor1).sum() + linear(tensor2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        # Get the current state of the optimizer
        state_dict = optimizer.state_dict()
    
        # Create a new optimizer
        new_optimizer = torch.optim.Adam([tensor1, tensor2], lr=0.01)
    
        # Load the state of the old optimizer into the new optimizer
        new_optimizer.load_state_dict(state_dict)
    
        # Perform one more optimization step with the new optimizer
        loss = linear(tensor1).sum() + linear(tensor2).sum()
        loss.backward()
        new_optimizer.step()
        
        # Return the updated tensor1 after loading the state dict
        return tensor1
    