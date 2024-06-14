import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.register_module_full_backward_pre_hook)
class TorchNnModulesModuleRegistermodulefullbackwardprehookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_full_backward_pre_hook_correctness(self):
        # Define a simple module
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super(SimpleModule, self).
                self.linear = torch.nn.Linear(10, 5)
    
            def forward(self, x):
                return self.linear(x)
    
        # Instantiate the module
        module = SimpleModule()
    
        # Define a backward pre-hook
        def backward_pre_hook(module, grad_output):
            # Return a new gradient with respect to the output
            new_grad_output = tuple(g * random.uniform(0.1, 2.0) if g is not None else None for g in grad_output)
            return new_grad_output
    
        # Register the backward pre-hook
        handle = torch.nn.modules.module.register_module_full_backward_pre_hook(backward_pre_hook)
    
        # Generate random input tensor
        input_tensor = torch.randn(10, 10)
    
        # Forward pass
        output = module(input_tensor)
    
        # Compute loss
        loss = output.sum()
    
        # Backward pass
        loss.backward()
    
        # Remove the hook
        handle.remove()
    
        return module.linear.weight.grad
    
    
    
    