import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.registermodulemoduleregistrationhook)
class TorchNnModulesModuleRegistermodulemoduleregistrationhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_module_registration_hook_correctness(self):
        # Define a module registration hook
        def module_registration_hook(module, name, submodule):
            # Modify the submodule, e.g., change its weight
            if submodule is not None:
                submodule.weight.data.fill_(1)
            return submodule
    