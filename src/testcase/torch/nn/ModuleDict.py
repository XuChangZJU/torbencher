import torch
import torch.nn as nn
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ModuleDict)
class TorchNnModuledictTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_moduledict_correctness(self):
        # Randomly generate input tensor dimensions
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 10)
        height = random.randint(5, 10)
        width = random.randint(5, 10)
        input_tensor = torch.randn(batch_size, channels, height, width)
    
        # Define a ModuleDict with random modules
        module_dict = nn.ModuleDict({
            'conv': nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            'pool': nn.MaxPool2d(kernel_size=2),
            'lrelu': nn.LeakyReLU(),
            'prelu': nn.PReLU()
        })
    
        # Randomly select a module from the ModuleDict
        choice = random.choice(['conv', 'pool'])
        act = random.choice(['lrelu', 'prelu'])
    
        # Apply the selected modules
        x = module_dict[choice](input_tensor)
        result = module_dict[act](x)
        
        return result
    
    
    
    