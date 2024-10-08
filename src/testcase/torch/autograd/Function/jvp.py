import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.Function.jvp)
class TorchAutogradFunctionJvpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_jvp_correctness(self):
        # Define the dimensions for the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensors
        input_tensor = torch.randn(input_size, requires_grad=True)
        tangent_tensor = torch.randn(input_size)

        # Define a custom function to override jvp
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                ctx.save_for_backward(input_tensor)
                return input_tensor * 2

            @staticmethod
            def backward(ctx, grad_output):
                input_tensor, = ctx.saved_tensors
                # Compute the gradient of the output with respect to the input
                grad_input = grad_output * 2
                return grad_input

            @staticmethod
            def jvp(ctx, input_jvp):
                input_tensor, = ctx.saved_tensors
                return input_jvp * input_tensor

        # Apply the custom function
        output_tensor = CustomFunction.apply(input_tensor)

        # Calculate jvp using PyTorch's autograd
        jvp_output = torch.autograd.functional.jvp(CustomFunction.apply, (input_tensor,), (tangent_tensor,))

        return jvp_output
