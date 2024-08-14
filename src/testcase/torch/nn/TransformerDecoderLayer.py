import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.TransformerDecoderLayer)
class TorchNnTransformerdecoderlayerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_transformer_decoder_layer_correctness(self):
        # Randomly generate parameters for TransformerDecoderLayer
        d_model = random.randint(128, 512)  # Random d_model between 128 and 512
        nhead = 1  # Random number of heads between 1 and 8
        dim_feedforward = random.randint(512, 2048)  # Random feedforward dimension between 512 and 2048

        # Create the TransformerDecoderLayer with random parameters
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)

        # Randomly generate sizes for tgt and memory tensors
        tgt_seq_len = random.randint(5, 20)  # Random target sequence length between 5 and 20
        memory_seq_len = random.randint(5, 20)  # Random memory sequence length between 5 and 20
        batch_size = random.randint(1, 10)  # Random batch size between 1 and 10

        # Generate random tgt and memory tensors
        tgt = torch.randn(tgt_seq_len, batch_size, d_model)
        memory = torch.randn(memory_seq_len, batch_size, d_model)

        # Pass the tensors through the decoder layer
        output = decoder_layer(tgt, memory)
        return output
