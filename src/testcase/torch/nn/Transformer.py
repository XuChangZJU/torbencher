
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Transformer)
class TorchNNTransformerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transformer(self):
        src = torch.randn(10, 32, 512)
        tgt = torch.randn(20, 32, 512)
        transformer = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048)
        result = transformer(src, tgt)
        return result

