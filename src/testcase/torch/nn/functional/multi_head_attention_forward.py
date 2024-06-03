
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.multi_head_attention_forward)
class TorchNNFunctionalMultiHeadAttentionForwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multi_head_attention_forward_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.multi_head_attention_forward(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9], input[10], input[11], need_weights=input[12])
            return result
        a = torch.randn(3, 2, 5)
        b = torch.randn(3, 2, 5)
        c = torch.randn(3, 2, 5)
        d = 5
        e = 3
        f = torch.nn.Linear(5, 5)
        g = torch.nn.Linear(5, 5)
        h = torch.nn.Linear(5, 5)
        i = torch.nn.Linear(5, 5)
        j = torch.nn.Linear(5, 5)
        k = None
        l = 0.5
        m = False
        result = torch.nn.functional.multi_head_attention_forward(a, b, c, d, e, f, g, h, i, j, k, l, need_weights=m)
        return result


