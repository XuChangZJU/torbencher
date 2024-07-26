import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Transformer)
class TorchNnTransformerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transformer_correctness(self):
        # Randomly generate parameters for the Transformer model
        d_model = random.randint(128, 512)  # Number of expected features in the encoder/decoder inputs
        nhead = 1  # Number of heads in the multiheadattention models
        num_encoder_layers = random.randint(1, 6)  # Number of sub-encoder-layers in the encoder
        num_decoder_layers = random.randint(1, 6)  # Number of sub-decoder-layers in the decoder
        dim_feedforward = random.randint(512, 2048)  # Dimension of the feedforward network model
        dropout = random.uniform(0.0, 0.3)  # Dropout value

        # Create the Transformer model with the generated parameters
        transformer_model = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Randomly generate input tensors for the source and target sequences
        src_seq_len = random.randint(5, 10)  # Length of the source sequence
        tgt_seq_len = random.randint(5, 10)  # Length of the target sequence
        batch_size = random.randint(1, 4)  # Batch size

        src = torch.randn((src_seq_len, batch_size, d_model))  # Source tensor
        tgt = torch.randn((tgt_seq_len, batch_size, d_model))  # Target tensor

        # Pass the source and target tensors through the Transformer model
        output = transformer_model(src, tgt)
        return output
