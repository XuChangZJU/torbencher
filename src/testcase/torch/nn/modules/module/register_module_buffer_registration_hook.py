import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.modules.module.register_module_buffer_registration_hook)
class TorchNnModulesModuleRegisterUmoduleUbufferUregistrationUhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_buffer_registration_hook_correctness(self):
        # Define a hook function that modifies the buffer
        def hook_fn(module, name, buffer):
            return buffer * 2

        # Register the hook
        hook_handle = torch.nn.modules.module.register_module_buffer_registration_hook(hook_fn)

        # Create a module and register a buffer
        module = torch.nn.Module()
        buffer_data = torch.randn(3, 4)
        module.register_buffer('my_buffer', buffer_data)

        # Check that the buffer has been modified by the hook
        assert torch.allclose(module.my_buffer, buffer_data * 2)

        # Remove the hook
        hook_handle.remove()

        # Register a new buffer and check that it is not modified by the hook
        new_buffer_data = torch.randn(2, 5)
        module.register_buffer('new_buffer', new_buffer_data)
        assert torch.allclose(module.new_buffer, new_buffer_data)

        return module.new_buffer
