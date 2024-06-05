
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Transformer)
class TorchTransformerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transformer_correctness(self):
        d_model = random.randint(1, 10)
        nhead = random.randint(1, 10)
        num_encoder_layers = random.randint(1, 10)
        num_decoder_layers = random.randint(1, 10)
        dim_feedforward = random.randint(1, 10)
        dropout = random.uniform(0.0, 1.0)
        activation = 'relu'
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), d_model)
        tgt_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), d_model)
        memory_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), d_model)
        src_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        tgt_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        memory_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        src_key_padding_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        tgt_key_padding_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        memory_key_padding_mask = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        result = transformer(input_tensor, tgt_tensor, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_transformer_large_scale(self):
        d_model = random.randint(100, 1000)
        nhead = random.randint(10, 100)
        num_encoder_layers = random.randint(1, 10)
        num_decoder_layers = random.randint(1, 10)
        dim_feedforward = random.randint(100, 1000)
        dropout = random.uniform(0.0, 1.0)
        activation = 'relu'
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), d_model)
        tgt_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), d_model)
        memory_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), d_model)
        src_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000)), dtype=torch.bool)
        tgt_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000)), dtype=torch.bool)
        memory_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000)), dtype=torch.bool)
        src_key_padding_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000)), dtype=torch.bool)
        tgt_key_padding_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000)), dtype=torch.bool)
        memory_key_padding_mask = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000)), dtype=torch.bool)
        transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        result = transformer(input_tensor, tgt_tensor, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return result

