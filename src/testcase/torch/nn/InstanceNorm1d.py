import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.InstanceNorm1d)
class TorchNnInstancenorm1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm1d_correctness(self):
        # Randomly generate the number of features (channels)
        num_features = random.randint(1, 10)
        
        # Randomly decide whether to use affine parameters
        affine = random.choice([True, False])
        
        # Randomly decide whether to track running stats
        track_running_stats = random.choice([True, False])
        
        # Create the InstanceNorm1d layer with random parameters
        instance_norm = torch.nn.InstanceNorm1d(num_features, affine=affine, track_running_stats=track_running_stats)
        
        # Randomly generate the batch size and sequence length
        batch_size = random.randint(1, 5)
        seq_length = random.randint(10, 50)
        
        # Generate a random input tensor with shape (N, C, L)
        input_tensor = torch.randn(batch_size, num_features, seq_length)
        
        # Apply instance normalization
        output_tensor = instance_norm(input_tensor)
        
        return output_tensor
    