import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.register_module_forward_pre_hook)
class TorchNnModulesModuleRegistermoduleforwardprehookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_forward_pre_hook_correctness(self):
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x * 2
    
        def hook_fn(module, input):
            # Modify the input by adding a random tensor of the same shape
            input_size = input[0].shape
            random_tensor = torch.randn(input_size)
            return (input[0] + random_tensor,)
    
        # Create a random tensor input
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        input_tensor = torch.randn(input_size)
    
        # Instantiate the module and register the hook
        module = SimpleModule()
        handle = torch.nn.modules.module.register_module_forward_pre_hook(hook_fn)
    
        # Pass the input tensor through the module
        result = module(input_tensor)
    
        # Remove the hook
        handle.remove()
    
        return result
    
    
    
    