
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.batch_norm)
class TorchBatch_normTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_norm(self, input=None):
        if input is not None:
            result = torch.batch_norm(input[0], input[1], input[2], input[3], input[4], training=input[5])
            return result
        input = torch.randn(20, 100)
        mean = torch.randn(100)
        var = torch.randn(100)
        weight = torch.randn(100)
        bias = torch.randn(100)
        result = torch.batch_norm(input, mean, var, weight, bias, training=True)
        return result

