import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LayerNorm)
class TorchNnLayernormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_layernorm_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 10)  # Random batch size between 1 and 10
        num_features = random.randint(1, 10)  # Random number of features between 1 and 10
        feature_size = random.randint(1, 10)  # Random feature size between 1 and 10
    
        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, num_features, feature_size)
    
        # Randomly generate normalized_shape
        normalized_shape = [num_features, feature_size]
    
        # Create LayerNorm instance with the generated normalized_shape
        layer_norm = torch.nn.LayerNorm(normalized_shape)
    
        # Apply LayerNorm to the input tensor
        result = layer_norm(input_tensor)
        return result
    
    
    
    