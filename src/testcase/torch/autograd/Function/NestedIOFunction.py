import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.function.NestedIOFunction)
class TorchAutogradFunctionNestediofunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nestediofunction_correctness(self):
        # Randomly generate input sizes for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        tensor1 = torch.randn(input_size, requires_grad=True)
        tensor2 = torch.randn(input_size, requires_grad=True)

        # Define a custom function using NestedIOFunction
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input1, input2):
                ctx.save_for_backward(input1, input2)
                return input1 + input2, input1 * input2

            @staticmethod
            def backward(ctx, grad_output1, grad_output2):
                input1, input2 = ctx.saved_tensors
                grad_input1 = grad_output1 + grad_output2 * input2
                grad_input2 = grad_output2 + grad_output1 * input1
                return grad_input1, grad_input2

        # Apply the custom function
        output1, output2 = MyFunction.apply(tensor1, tensor2)

        # Calculate gradients
        grad_output1 = torch.randn(input_size)
        grad_output2 = torch.randn(input_size)
        output1.backward(grad_output1, retain_graph=True)
        output2.backward(grad_output2)

        # Return the gradients of the inputs
        return tensor1.grad, tensor2.grad
