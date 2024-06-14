import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.registermoduleparameterregistrationhook)
class TorchNnModulesModuleRegistermoduleparameterregistrationhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_parameter_registration_hook_correctness(self):
        # Define a hook function
        def hook_fn(module, name, param):
            # Modify the parameter (e.g., scale it by 2)
            return 2 * param
    