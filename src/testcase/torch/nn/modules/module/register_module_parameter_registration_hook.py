import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.modules.module.register_module_parameter_registration_hook)
class TorchNnModulesModuleRegistermoduleparameterregistrationhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_parameter_registration_hook_correctness(self):
        # Define a hook function
        def hook_fn(module, name, param):
            # Modify the parameter (e.g., scale it by 2)
            return 2 * param

        # Create a module
        module = torch.nn.Module()

        # Register the hook
        handle = torch.nn.modules.module.register_module_parameter_registration_hook(hook_fn)

        # Add a parameter to the module
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        param = torch.nn.Parameter(torch.randn(input_size))
        module.register_parameter('test_param', param)

        # Check if the hook modified the parameter
        result = module.test_param
        handle.remove()

        return result
