import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.InstanceNorm2d)
class TorchNnInstancenorm2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm2d_correctness(self):
        # Randomly generate the number of features (channels)
        num_features = random.randint(1, 10)
        
        # Randomly generate the batch size, height, and width
        batch_size = random.randint(1, 5)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        
        # Create a random input tensor with shape (N, C, H, W)
        input_tensor = torch.randn(batch_size, num_features, height, width)
        
        # Create an InstanceNorm2d layer with the generated number of features
        instance_norm = torch.nn.InstanceNorm2d(num_features)
        
        # Apply the instance normalization to the input tensor
        output_tensor = instance_norm(input_tensor)
        
        return output_tensor
    