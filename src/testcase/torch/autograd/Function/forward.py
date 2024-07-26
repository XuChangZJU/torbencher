import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.Function.forward)
class TorchAutogradFunctionForwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_autograd_function_forward_correctness(self):
        # Define a custom autograd function
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight):
                ctx.save_for_backward(input, weight)
                return input @ weight
    
            @staticmethod
            def backward(ctx, grad_output):
                input, weight = ctx.saved_tensors
                grad_input = grad_weight = None
                if ctx.needs_input_grad[0]:
                    grad_input = grad_output @ weight.t()
                if ctx.needs_input_grad[1]:
                    grad_weight = input.t() @ grad_output
                return grad_input, grad_weight
    
        # Generate random input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input = torch.randn(input_size, requires_grad=True)
    
        # Generate random weight tensor
        weight_size = [input_size[-1], random.randint(1, 5)]
        weight = torch.randn(weight_size, requires_grad=True)
    
        # Apply the custom function
        output = MyFunction.apply(input, weight)
        
        return output
    