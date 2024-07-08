import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.Function.backward)
class TorchAutogradFunctionBackwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_autograd_function_backward_correctness(self):
        # Define the size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random input tensor that requires gradient
        input_tensor = torch.randn(input_size, requires_grad=True)
    
        # Define a custom function
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # Save the input tensor in the context
                ctx.save_for_backward(input)
                # Perform some operation on the input
                output = input * 2
                return output
    
            @staticmethod
            def backward(ctx, grad_output):
                # Retrieve the saved input tensor from the context
                input, = ctx.saved_tensors
                # Compute the gradient w.r.t. the input
                grad_input = grad_output * 2
                return grad_input
    
        # Apply the custom function to the input tensor
        output_tensor = MyFunction.apply(input_tensor)
    
        # Compute the gradients
        output_tensor.backward(torch.ones_like(output_tensor))
    
        # Return the gradient of the input tensor
        return input_tensor.grad
    