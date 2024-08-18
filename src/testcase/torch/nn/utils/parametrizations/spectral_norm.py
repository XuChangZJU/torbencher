import random
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrize

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

import unittest
@test_api(parametrize.spectral_norm)
class TorchNnUtilsParametrizationsSpectralUnormTestCase(TorBencherTestCaseBase):
    @unittest.skip
    @test_api_version.larger_than("2.0.0")
    def test_spectral_norm_correctness(self):
        # Randomly generate dimensions for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a Linear layer with random dimensions
        linear_layer = nn.Linear(in_features, out_features)

        # Initialize the weights and biases of the linear layer
        with torch.no_grad():
            linear_layer.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)
            if linear_layer.bias is not None:
                linear_layer.bias = torch.nn.Parameter(torch.randn(out_features) * 0.01)

        # Apply spectral normalization to the Linear layer
        parametrize.spectral_norm(linear_layer)

        # Generate a random input tensor with appropriate dimensions
        input_tensor = torch.randn(random.randint(1, 5), in_features)

        # Forward pass through the spectrally normalized layer
        output = linear_layer(input_tensor)

        return output
