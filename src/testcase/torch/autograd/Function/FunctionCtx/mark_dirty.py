import torch
import random
from torch.autograd import Function


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.function.FunctionCtx.mark_dirty)
class TorchAutogradFunctionFunctionctxMarkdirtyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
        def forward(ctx, x):
            x_npy = x.detach().numpy()  # Detach to avoid in-place operation on leaf variable
            x_npy += 1
            ctx.mark_dirty(x)
            return x
    
        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            return grad_output
    
    def test_mark_dirty_correctness(self):
        # Randomly generate tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Create a random tensor with requires_grad=True
        tensor = torch.randn(input_size, requires_grad=True)
    
        # Apply the Inplace function
        result = Inplace.apply(tensor)
    
        return result
    
    
    
    