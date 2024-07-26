import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.graph.register_multi_grad_hook)
class TorchAutogradGraphRegistermultigradhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_multi_grad_hook_correctness(self):
        # Define dimensions for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create tensors a and b with requires_grad=True
        a = torch.randn(input_size, requires_grad=True)
        b = torch.randn(input_size, requires_grad=True)
    
        # Compute tensors c and d based on a and b
        c = a * b
        d = a * b
    
        # Define a hook function to print gradient availability
        def fn(grads):
            [g is not None for g in grads]
            return grads
    
        # Register the hook for tensors a, b, c, and d
        handle = torch.autograd.graph.register_multi_grad_hook((a, b, c, d), fn)
    
        # Compute gradients for c.sum() and check gradient availability
        c.sum().backward(retain_graph=True)
    
        # Compute gradients for c.sum() with inputs=(a,) and check gradient availability
        c.sum().backward(inputs=(a,), retain_graph=True)
    
        # Remove the hook
        handle.remove()
    
        # Return the last computed gradients
        return a.grad
    