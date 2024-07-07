import torch
import random
from torch.autograd import Function


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.function.FunctionCtx.mark_non_differentiable)
class TorchAutogradFunctionFunctionctxMarknondifferentiableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
        def forward(ctx, x):
            sorted_tensor, indices = x.sort()
            ctx.mark_non_differentiable(indices)
            ctx.save_for_backward(x, indices)
            return sorted_tensor, indices
    
        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_sorted, grad_indices):
            x, indices = ctx.saved_tensors
            grad_input = torch.zeros_like(x)
            grad_input.scatter_(0, indices, grad_sorted)
            return grad_input, None
    
    def test_mark_non_differentiable_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size
    
        input_tensor = torch.randn(input_size, requires_grad=True)  # Random input tensor with gradient tracking
        sorted_tensor, indices = TestFunction.apply(input_tensor)  # Apply the custom function
    
        # Perform backward pass
        sorted_tensor.sum().
    
        # Check if gradients are correctly computed
        return input_tensor.grad, indices.requires_grad
    
    
    
    