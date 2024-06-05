
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MultiheadAttention)
class TorchMultiheadAttentionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multiheadattention_correctness(self):
        embed_dim = random.randint(1, 10)
        num_heads = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), embed_dim)
        key_padding_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        attn_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        result = multihead_attn(input_tensor, input_tensor, input_tensor, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_multiheadattention_large_scale(self):
        embed_dim = random.randint(100, 1000)
        num_heads = random.randint(10, 100)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), embed_dim)
        key_padding_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000)), dtype=torch.bool)
        attn_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000)), dtype=torch.bool)
        multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        result = multihead_attn(input_tensor, input_tensor, input_tensor, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return result

