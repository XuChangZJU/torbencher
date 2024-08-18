import random

import torch
import torch.nn as nn

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest


@test_api(torch.nn.ModuleDict)
class TorchNnModuledictTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    # @unittest.skip
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
        with torch.no_grad():
            # Initialize Conv2d weights and biases
            conv_layer = module_dict['conv']
            conv_layer.weight = nn.Parameter(torch.normal(0, 0.01, conv_layer.weight.shape))
            if conv_layer.bias is not None:
                conv_layer.bias = nn.Parameter(torch.normal(0, 0.01, conv_layer.bias.shape))

            # Initialize PReLU weights
            prelu_layer = module_dict['prelu']
            prelu_layer.weight = nn.Parameter(torch.normal(0, 0.01, prelu_layer.weight.shape))

        # Randomly select a module from the ModuleDict
        choice = random.choice(['conv', 'pool'])
        act = random.choice(['lrelu', 'prelu'])

        # Apply the selected modules
        x = module_dict[choice](input_tensor)
        result = module_dict[act](x)

        return result
