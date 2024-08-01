import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.MultiheadAttention)
class TorchNnMultiheadattentionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nn_MultiheadAttention_correctness(self):
        # Define the parameters for the MultiheadAttention module
        embed_dim = 10  # Total dimension of the model
        num_heads = 5  # Number of parallel attention heads, should divide embed_dim
        dropout = random.uniform(0.0, 1.0)  # Dropout probability
        bias = random.choice([True, False])  # Whether to add bias to input/output projection layers
        add_bias_kv = random.choice([True, False])  # Whether to add bias to the key and value sequences
        add_zero_attn = random.choice(
            [True, False])  # Whether to add a new batch of zeros to the key and value sequences
        kdim = random.randint(1, 10) if random.choice([True, False]) else embed_dim  # Total number of features for keys
        vdim = random.randint(1, 10) if random.choice(
            [True, False]) else embed_dim  # Total number of features for values
        batch_first = random.choice(
            [True, False])  # Whether the input and output tensors are provided as (batch, seq, feature)

        # Define the input tensors
        if batch_first:
            batch_size = random.randint(1, 5)
            seq_len_q = random.randint(1, 10)
            seq_len_k = random.randint(1, 10)
            seq_len_v = seq_len_k
            query = torch.randn(batch_size, seq_len_q, embed_dim)
            key = torch.randn(batch_size, seq_len_k, kdim)
            value = torch.randn(batch_size, seq_len_v, vdim)
        else:
            seq_len_q = random.randint(1, 10)
            seq_len_k = random.randint(1, 10)
            seq_len_v = seq_len_k
            batch_size = random.randint(1, 5)
            query = torch.randn(seq_len_q, batch_size, embed_dim)
            key = torch.randn(seq_len_k, batch_size, kdim)
            value = torch.randn(seq_len_v, batch_size, vdim)

        # Create the MultiheadAttention module
        multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                                                     kdim, vdim, batch_first)

        # Perform the attention operation
        attn_output, attn_output_weights = multihead_attn(query, key, value)

        # Return the attention output
        return attn_output
