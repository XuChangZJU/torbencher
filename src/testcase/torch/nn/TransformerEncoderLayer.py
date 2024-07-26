import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.TransformerEncoderLayer)
class TorchNnTransformerencoderlayerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transformer_encoder_layer_correctness(self):
        # Randomly generate parameters for TransformerEncoderLayer
        d_model = random.randint(32, 512)  # Number of expected features in the input
        nhead = 1  # Number of heads in the multiheadattention models
        dim_feedforward = random.randint(512, 2048)  # Dimension of the feedforward network model
        dropout = random.uniform(0.0, 0.5)  # Dropout value
        layer_norm_eps = random.uniform(1e-6, 1e-4)  # Eps value in layer normalization components
        batch_first = random.choice(
            [True, False])  # Whether input and output tensors are provided as (batch, seq, feature)
        norm_first = random.choice(
            [True, False])  # Whether layer norm is done prior to attention and feedforward operations

        # Create the TransformerEncoderLayer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, 'relu',
                                                         layer_norm_eps, batch_first, norm_first)

        # Randomly generate input tensor
        batch_size = random.randint(1, 10)  # Random batch size
        seq_length = random.randint(5, 20)  # Random sequence length
        if batch_first:
            src = torch.randn(batch_size, seq_length, d_model)  # Input tensor with shape (batch, seq, feature)
        else:
            src = torch.randn(seq_length, batch_size, d_model)  # Input tensor with shape (seq, batch, feature)

        # Forward pass through the encoder layer
        result = encoder_layer(src)
        return result
