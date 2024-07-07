import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.function.FunctionCtx.save_for_backward)
class TorchAutogradFunctionFunctionctxSaveforbackwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_save_for_backward_correctness(self):
        # Define a custom function to test save_for_backward
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                # Save tensors for backward
                ctx.save_for_backward(x, y)
                return x + y
    
            @staticmethod
            def backward(ctx, grad_output):
                # Retrieve saved tensors
                x, y = ctx.saved_tensors
                return grad_output, grad_output
    
        # Generate random input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size, requires_grad=True)
        y = torch.randn(input_size, requires_grad=True)
    
        # Apply the custom function
        output = MyFunc.apply(x, y)
    
        # Compute gradients
        output.backward(torch.ones_like(output))
    
        # Return the gradients of x and y
        return x.grad, y.grad
    
    
    
    