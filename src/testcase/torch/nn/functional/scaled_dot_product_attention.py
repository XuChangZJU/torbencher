
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.scaled_dot_product_attention)
class ScaledDotProductAttentionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scaled_dot_product_attention_correctness(self):
        batch_size = random.randint(1, 10)
        num_heads = random.randint(1, 10)
        query_length = random.randint(10, 20)
        key_length = random.randint(10, 20)
        value_length = random.randint(10, 20)
        embed_dim = random.randint(10, 20)
        query = torch.randn(batch_size, num_heads, query_length, embed_dim)
        key = torch.randn(batch_size, num_heads, key_length, embed_dim)
        value = torch.randn(batch_size, num_heads, value_length, embed_dim)
        attn_mask = torch.randint(0, 2, (batch_size, num_heads, query_length, key_length))
        dropout_p = random.uniform(0.0, 1.0)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_scaled_dot_product_attention_large_scale(self):
        batch_size = random.randint(100, 1000)
        num_heads = random.randint(100, 1000)
        query_length = random.randint(1000, 2000)
        key_length = random.randint(1000, 2000)
        value_length = random.randint(1000, 2000)
        embed_dim = random.randint(100, 1000)
        query = torch.randn(batch_size, num_heads, query_length, embed_dim)
        key = torch.randn(batch_size, num_heads, key_length, embed_dim)
        value = torch.randn(batch_size, num_heads, value_length, embed_dim)
        attn_mask = torch.randint(0, 2, (batch_size, num_heads, query_length, key_length))
        dropout_p = random.uniform(0.0, 1.0)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p)
        return result

