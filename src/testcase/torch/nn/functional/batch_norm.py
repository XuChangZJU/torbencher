
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.batch_norm)
class TorchNNFunctionalBatchNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_norm_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.batch_norm(input[0], input[1], input[2], input[3], input[4], training=input[5], momentum=input[6], eps=input[7])
            return [result, input]
        a = torch.randn(20, 100)
        b = torch.ones(100)
        c = torch.ones(100)
        d = torch.ones(100)
        e = torch.ones(100)
        f = True
        g = 0.1
        h = 1e-05
        result = torch.nn.functional.batch_norm(a, b, c, d, e, training=f, momentum=g, eps=h)
        return [result, [a, b, c, d, e, f, g, h]]


