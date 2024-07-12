import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


class CustomClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Clamp the input between -1 and 1
        return torch.clamp(input, min=-1, max=1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Gradient is 1 for inputs in the interval [-1, 1], 0 otherwise
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


@test_api(CustomClampFunction.apply)
class TorchAutogradFunctionOncedifferentiableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_once_differentiable_correctness(self):
        # Randomly generate input tensor x
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size, requires_grad=True)
    
        # Apply the custom function
        result = CustomClampFunction.apply(x)
    
        # Calculate gradients
        result.sum().backward()
    
        # Return the result and the gradients
        return result, x.grad
    