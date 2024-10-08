import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.TransformerEncoder)
class TorchNnTransformerencoderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_transformer_encoder_correctness(self):
        # Randomly generate parameters for TransformerEncoderLayer
        d_model = random.randint(128, 1024)  # Random model dimension between 128 and 1024
        nhead = 1  # Random number of attention heads between 1 and 16
        num_layers = random.randint(1, 12)  # Random number of encoder layers between 1 and 12

        # Create an instance of TransformerEncoderLayer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        # Create an instance of TransformerEncoder
        transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Randomly generate input tensor dimensions
        seq_length = random.randint(5, 20)  # Random sequence length between 5 and 20
        batch_size = random.randint(1, 32)  # Random batch size between 1 and 32
        src = torch.rand(seq_length, batch_size, d_model)  # Random input tensor

        # Pass the input tensor through the transformer encoder
        output = transformer_encoder(src)
        return output
