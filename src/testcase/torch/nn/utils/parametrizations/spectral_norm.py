import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrize
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.parametrizations.spectral_norm)
class TorchNnUtilsParametrizationsSpectralnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_spectral_norm_correctness(self):
        # Randomly generate dimensions for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        
        # Create a Linear layer with random dimensions
        linear_layer = nn.Linear(in_features, out_features)
        
        # Apply spectral normalization to the Linear layer
        snm = parametrize.spectral_norm(linear_layer)
        
        # Generate a random input tensor with appropriate dimensions
        input_tensor = torch.randn(random.randint(1, 5), in_features)
        
        # Forward pass through the spectrally normalized layer
        output = snm(input_tensor)
        
        return output
    
    
    
    