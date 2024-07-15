import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.register_module_backward_hook)
class TorchNnModulesModuleRegistermodulebackwardhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_backward_hook_correctness(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super(SimpleModule, self).__init__()
                self.linear = torch.nn.Linear(random.randint(1, 10), random.randint(1, 10))
    
            def forward(self, x):
                return self.linear(x)
    
        def backward_hook(module, grad_input, grad_output):
            # Modify the gradients for testing purposes
            return tuple(g * random.uniform(0.1, 2.0) for g in grad_input)
    
        # Create a random input tensor
        input_tensor = torch.randn(random.randint(1, 5), random.randint(1, 10))
    
        # Initialize the module and register the backward hook
        module = SimpleModule()
        handle = module.register_module_backward_hook(backward_hook)
    
        # Forward pass
        output = module(input_tensor)
    
        # Backward pass
        output.sum().backward()
    
        # Remove the hook
        handle.remove()
    
        return module.linear.weight.grad
    
    
    
    