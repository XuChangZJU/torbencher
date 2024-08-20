import torch
import torch.nn as nn
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Transformer)
class TorchNnTransformerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_transformer_correctness(self):
        transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((20, 32, 512))
        out = transformer_model(src, tgt)
        return out
