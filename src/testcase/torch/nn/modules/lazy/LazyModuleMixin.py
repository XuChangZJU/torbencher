import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.modules.lazy.LazyModuleMixin)
class TorchNnModulesLazyLazymodulemixinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_module_mixin_correctness(self):
        class LazyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.LazyLinear(random.randint(5, 15))  # Random out_features between 5 and 15
                self.relu1 = torch.nn.ReLU()
                self.fc2 = torch.nn.LazyLinear(random.randint(1, 5))  # Random out_features between 1 and 5
                self.relu2 = torch.nn.ReLU()
    
            def forward(self, input):
                x = self.relu1(self.fc1(input))
                y = self.relu2(self.fc2(x))
                return y
    