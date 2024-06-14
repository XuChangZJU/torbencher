import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.registermoduleforwardhook)
class TorchNnModulesModuleRegistermoduleforwardhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_forward_hook_correctness(self):
        # Define a simple module
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x * 2
    