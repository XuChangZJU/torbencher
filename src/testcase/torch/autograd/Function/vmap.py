import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.Function.vmap)
class TorchAutogradFunctionVmapTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vmap_correctness(self):
        # Randomly generate batch size for vmap
        batch_size = random.randint(1, 4)

        # Randomly generate dimension and size for input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensors
        tensor1 = torch.randn([batch_size] + input_size)
        tensor2 = torch.randn([batch_size] + input_size)

        # Define a custom autograd function
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input1, input2):
                ctx.save_for_backward(input1, input2)
                return input1 + input2

            @staticmethod
            def backward(ctx, grad_output):
                input1, input2 = ctx.saved_tensors
                return grad_output, grad_output

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(*inputs)

        # Apply vmap to the custom function
        result = torch.vmap(MyFunction.apply, in_dims=(0, 0))(tensor1, tensor2)
        return result
