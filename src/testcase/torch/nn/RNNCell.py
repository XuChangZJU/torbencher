import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNNCell)
class TorchNnRnncellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnncell_correctness(self):
    # Define input size, hidden size and batch size
    input_size = random.randint(1, 10)
    hidden_size = random.randint(1, 10)
    batch_size = random.randint(1, 10)
    # Define sequence length
    seq_len = random.randint(1, 10)

    # Create input tensor
    input_tensor = torch.randn(seq_len, batch_size, input_size)

    # Create hidden tensor
    hidden_tensor = torch.randn(batch_size, hidden_size)

    # Create RNNCell
    rnn_cell = torch.nn.RNNCell(input_size, hidden_size)

    # Forward pass
    output_list = []
    for i in range(seq_len):
        hidden_tensor = rnn_cell(input_tensor[i], hidden_tensor)
        output_list.append(hidden_tensor)

    # Concatenate output
    output_tensor = torch.stack(output_list, dim=0)
    
    return output_tensor
