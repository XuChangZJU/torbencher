
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Transformer)
class TorchNNTransformerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transformer(self, input=None):
        if input is not None:
            result = torch.nn.Transformer(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9], input[10], input[11])(input[12], input[13])
            return [result, input]
        src = torch.randn(10, 32, 512)
        tgt = torch.randn(20, 32, 512)
        transformer = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048)
        result = transformer(src, tgt)
        return [result, [10, 32, 512, 20, 32, 512, 512, 8, 6, 6, 2048, 0.1, src, tgt]]

