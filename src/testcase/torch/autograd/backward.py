import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.backward)
class TorchAutogradBackwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_backward_correctness(self):
        # Randomly generate tensor dimensions and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random tensor with requires_grad=True
        tensor = torch.randn(input_size, requires_grad=True)
    
        # Compute a scalar output from the tensor
        output = (tensor * tensor).sum()
    
        # Call backward to compute gradients
        output.backward()
    
        # Return the gradient of the tensor
        return tensor.grad
    