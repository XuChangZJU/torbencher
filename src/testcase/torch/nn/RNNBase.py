import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNNBase)
class TorchNnRnnbaseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnnbase_correctness(self):
        input_size = random.randint(1, 10)  # Random input size
        hidden_size = random.randint(1, 10)  # Random hidden size
        num_layers = random.randint(1, 3)  # Random number of layers
        batch_size = random.randint(1, 5)  # Random batch size
        seq_length = random.randint(1, 5)  # Random sequence length
    
        # Create random input tensor with shape (seq_length, batch_size, input_size)
        input_tensor = torch.randn(seq_length, batch_size, input_size)
        
        # Initialize RNNBase with random parameters
        rnn = torch.nn.RNNBase(mode='RNN_TANH', input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        
        # Initialize hidden state with shape (num_layers, batch_size, hidden_size)
        h_0 = torch.randn(num_layers, batch_size, hidden_size)
        
        # Perform forward pass
        output, h_n = rnn(input_tensor, h_0)
        
        return output, h_n
    
    
    
    