import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.module.registermodulefullbackwardhook)
class TorchNnModulesModuleRegistermodulefullbackwardhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_module_full_backward_hook_correctness(self):
        # Define a simple neural network module
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 2)
    
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
    