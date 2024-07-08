import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.function.once_differentiable)
class TorchAutogradFunctionOncedifferentiableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_once_differentiable_correctness(self):
        # Randomly generate input tensor x
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size, requires_grad=True)
    
        # Define a custom function with once_differentiable
        @torch.autograd.function.once_differentiable
        def my_custom_function(input):
            # Define the forward pass with an operation that is only differentiable once
            output = input ** 2
            return output
    
        # Apply the custom function
        result = my_custom_function(x)
    
        # Calculate gradients
        result.sum().backward()
    
        # Return the result and the gradients
        return result, x.grad
    