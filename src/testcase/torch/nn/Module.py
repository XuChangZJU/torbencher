import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Module)
class TorchNnModuleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nn_module_correctness(self):
        class RandomModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Randomly generate the number of input and output channels for Conv2d layers
                in_channels1 = random.randint(1, 10)
                out_channels1 = random.randint(1, 10)
                out_channels2 = random.randint(1, 10)
                kernel_size = random.randint(1, 5)
                
                self.conv1 = torch.nn.Conv2d(in_channels1, out_channels1, kernel_size)
                self.conv2 = torch.nn.Conv2d(out_channels1, out_channels2, kernel_size)
    
            def forward(self, x):
                x = torch.nn.functional.relu(self.conv1(x))
                return torch.nn.functional.relu(self.conv2(x))
    