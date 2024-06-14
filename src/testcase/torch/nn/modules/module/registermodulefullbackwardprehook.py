import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.registermodulefullbackwardprehook)
class TorchNnModulesModuleRegistermodulefullbackwardprehookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_full_backward_pre_hook_correctness(self):
        # Define a simple module
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super(SimpleModule, self).__init__()
                self.linear = torch.nn.Linear(10, 5)
    
            def forward(self, x):
                return self.linear(x)
    