import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.function.InplaceFunction)
class TorchAutogradFunctionInplacefunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inplace_function_correctness(self):
        # Generate random dimensions for the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random input tensor
        input_tensor = torch.randn(input_size, requires_grad=True)
    
        # Define a custom inplace function
        class MyInplaceFunction(torch.autograd.function.InplaceFunction):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.mul_(2)  # Example inplace operation: multiply by 2
    
            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                return grad_output.mul(2)  # Gradient computation for the inplace operation
    
        # Apply the custom inplace function
        output_tensor = MyInplaceFunction.apply(input_tensor)
    
        # Compute gradients
        output_tensor.sum().
    
        # Return the output tensor and the gradient of the input tensor
        return output_tensor, input_tensor.grad
    
    # Automatically added function calls
    
    
    