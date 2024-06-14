import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Optimizer.addparamgroup)
class TorchOptimOptimizerAddparamgroupTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_param_group_correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a tensor
        tensor = torch.randn(input_size, requires_grad=True)
    
        # Create an optimizer
        optimizer = torch.optim.SGD([tensor], lr=0.1)
    
        # Define a new parameter group
        param_group = {
            'params': [torch.randn(input_size, requires_grad=True)],  # New tensor to optimize
            'lr': random.uniform(0.01, 0.1)  # Different learning rate for the new group
        }
    
        # Add the new parameter group to the optimizer
        result = optimizer.add_param_group(param_group)
    
        return result
    