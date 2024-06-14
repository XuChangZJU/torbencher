import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyInstanceNorm1d)
class TorchNnLazyinstancenorm1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_instance_norm_1d_correctness(self):
        # Randomly choose the batch size (N) and sequence length (L)
        batch_size = random.randint(1, 4)
        seq_length = random.randint(1, 10)
        
        # Randomly choose the number of channels (C)
        num_channels = random.randint(1, 5)
        
        # Create a random input tensor of shape (N, C, L)
        input_tensor = torch.randn(batch_size, num_channels, seq_length)
        
        # Initialize LazyInstanceNorm1d without specifying num_features
        lazy_instance_norm = torch.nn.LazyInstanceNorm1d()
        
        # Apply the LazyInstanceNorm1d to the input tensor
        result = lazy_instance_norm(input_tensor)
        
        return result
    