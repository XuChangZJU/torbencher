import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.graph.get_gradient_edge)
class TorchAutogradGraphGetgradientedgeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_gradient_edge_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Randomly generate input tensor
        input = torch.randn(input_size, requires_grad=True)
        
        # Calculate loss
        loss = torch.sum(input * input)
        
        # Get gradient edge
        gradient_edge = torch.autograd.graph.get_gradient_edge(input)
        
        # Calculate gradient using gradient edge
        gradient = torch.autograd.grad(loss, gradient_edge)
        
        return gradient
    