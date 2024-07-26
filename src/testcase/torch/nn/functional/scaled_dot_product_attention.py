import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.scaled_dot_product_attention)
class TorchNnFunctionalScaleddotproductattentionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scaled_dot_product_attention_correctness(self):
        # Random shapes for query, key, and value tensors
        batch_size = random.randint(1, 4)
        src_sequence_length = random.randint(1, 10)
        tgt_sequence_length = random.randint(1, 10)
        embedding_dim = random.randint(1, 32)
        value_emb_dim = random.randint(1, 32)
    
        # Generate random tensors for query, key, and value
        query = torch.randn(batch_size, tgt_sequence_length, embedding_dim)
        key = torch.randn(batch_size, src_sequence_length, embedding_dim)
        value = torch.randn(batch_size, src_sequence_length, value_emb_dim)
        
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return result
    
    
    
    