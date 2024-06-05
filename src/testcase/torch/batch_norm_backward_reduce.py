
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.batch_norm_backward_reduce)
class TorchBatchNormBackwardReduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_norm_backward_reduce_correctness(self):
        batch = random.randint(1, 10)
        channel = random.randint(1, 10)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        grad_output = torch.randn(batch, channel, height, width)
        input = torch.randn(batch, channel, height, width)
        running_mean = torch.randn(channel)
        running_var = torch.randn(channel)
        weight = torch.randn(channel)
        bias = torch.randn(channel)
        eps = random.uniform(0.1, 10.0)
        result = torch.batch_norm_backward_reduce(grad_output, input, running_mean, running_var, weight, bias, eps=eps)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_batch_norm_backward_reduce_large_scale(self):
        batch = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        height = random.randint(100, 1000)
        width = random.randint(100, 1000)
        grad_output = torch.randn(batch, channel, height, width)
        input = torch.randn(batch, channel, height, width)
        running_mean = torch.randn(channel)
        running_var = torch.randn(channel)
        weight = torch.randn(channel)
        bias = torch.randn(channel)
        eps = random.uniform(0.1, 10.0)
        result = torch.batch_norm_backward_reduce(grad_output, input, running_mean, running_var, weight, bias, eps=eps)
        return result

