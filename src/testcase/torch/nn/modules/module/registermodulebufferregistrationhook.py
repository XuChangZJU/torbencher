import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.registermodulebufferregistrationhook)
class TorchNnModulesModuleRegistermodulebufferregistrationhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_buffer_registration_hook_correctness(self):
    # Define a hook function that modifies the buffer
    def hook_fn(module, name, buffer):
        return buffer * 2
