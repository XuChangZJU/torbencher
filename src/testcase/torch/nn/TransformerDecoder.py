import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.TransformerDecoder)
class TorchNnTransformerdecoderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_transformer_decoder_correctness(self):
        # Randomly generate dimensions for the model
        d_model = random.randint(128, 1024)  # Random model dimension between 128 and 1024
        nhead = 1  # Random number of attention heads between 1 and 16
        num_layers = random.randint(1, 6)  # Random number of decoder layers between 1 and 6

        # Create a TransformerDecoderLayer instance
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)

        # Create a TransformerDecoder instance
        transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Randomly generate dimensions for the input tensors
        memory_seq_len = random.randint(5, 20)  # Random memory sequence length between 5 and 20
        tgt_seq_len = random.randint(5, 20)  # Random target sequence length between 5 and 20
        batch_size = random.randint(1, 32)  # Random batch size between 1 and 32

        # Generate random input tensors
        memory = torch.randn(memory_seq_len, batch_size, d_model)
        tgt = torch.randn(tgt_seq_len, batch_size, d_model)

        # Pass the tensors through the transformer decoder
        result = transformer_decoder(tgt, memory)
        return result
