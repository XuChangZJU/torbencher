import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest


@test_api(torch.nn.Transformer)
class TorchNnTransformerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_transformer_correctness(self):
        # torch.manual_seed(0) #手动设置随机种子可以通过，但是不手动设置就通过不了，应该是框架没有处理好，先不加，等框架处理
        # Randomly generate parameters for the Transformer model
        d_model = 512  # Number of expected features in the encoder/decoder inputs
        nhead = 8  # Number of heads in the multiheadattention models
        num_encoder_layers = random.randint(1, 6)  # Number of sub-encoder-layers in the encoder
        num_decoder_layers = random.randint(1, 6)  # Number of sub-decoder-layers in the decoder
        dim_feedforward = random.randint(512, 2048)  # Dimension of the feedforward network model

        # Create the Transformer model with the generated parameters
        transformer_model = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.0
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
