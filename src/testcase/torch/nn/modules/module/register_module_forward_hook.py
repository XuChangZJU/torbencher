import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.modules.module.register_module_forward_hook)
class TorchNnModulesModuleRegistermoduleforwardhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_forward_hook_correctness(self):
        # Define a simple module
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        # Instantiate the module
        module = SimpleModule()

        # Define a hook function
        def hook_fn(module, input, output):
            return output + 1

        # Register the hook
        handle = torch.nn.modules.module.register_module_forward_hook(hook_fn)

        # Generate random input tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        input_tensor = torch.randn(input_size)

        # Pass the input tensor through the module
        result = module(input_tensor)

        # Remove the hook
        handle.remove()

        return result
