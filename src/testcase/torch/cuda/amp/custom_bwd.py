import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.amp.custom_bwd)
class TorchCudaAmpCustombwdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_custom_bwd_correctness(self):
        class MyFunction(torch.autograd.Function):
            @staticmethod
            @torch.cuda.amp.custom_fwd
            def forward(ctx, input_tensor):
                ctx.save_for_backward(input_tensor)
                return input_tensor * 2

            @staticmethod
            @torch.cuda.amp.custom_bwd
            def backward(ctx, grad_output):
                input_tensor, = ctx.saved_tensors
                return grad_output * 2

        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size, requires_grad=True)

        # Apply the custom function
        output = MyFunction.apply(input_tensor)

        # Random gradient output tensor
        grad_output = torch.randn_like(output)

        # Perform backward pass
        output.backward(grad_output)

        return input_tensor.grad
