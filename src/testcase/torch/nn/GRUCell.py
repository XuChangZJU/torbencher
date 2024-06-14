import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GRUCell)
class TorchNnGrucellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_GRUCell_correctness(self):
    # Randomly generate input size
    input_size = random.randint(1, 10)
    # Randomly generate hidden size
    hidden_size = random.randint(1, 10)
    # Randomly generate batch size
    batch_size = random.randint(1, 10)
    # Randomly generate sequence length
    seq_len = random.randint(1, 10)

    # Create GRUCell module
    gru_cell = torch.nn.GRUCell(input_size, hidden_size)

    # Create random input tensor
    input_tensor = torch.randn(seq_len, batch_size, input_size)
    # Create random hidden tensor
    hidden_tensor = torch.randn(batch_size, hidden_size)

    # Iterate over the sequence
    for i in range(seq_len):
        # Calculate hidden state
        hidden_tensor = gru_cell(input_tensor[i], hidden_tensor)
    
    # Return the final hidden state
    return hidden_tensor
