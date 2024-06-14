import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.BatchNorm1d)
class TorchNnBatchnorm1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batchnorm1d_correctness(self):
    # Randomly generate the number of features (channels)
    num_features = random.randint(1, 10)
    
    # Randomly generate the batch size and sequence length
    batch_size = random.randint(1, 5)
    seq_length = random.randint(1, 5)
    
    # Create a random input tensor of shape (batch_size, num_features, seq_length)
    input_tensor = torch.randn(batch_size, num_features, seq_length)
    
    # Initialize BatchNorm1d with the randomly generated number of features
    batch_norm = torch.nn.BatchNorm1d(num_features)
    
    # Apply BatchNorm1d to the input tensor
    output_tensor = batch_norm(input_tensor)
    
    return output_tensor
