import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.batch_norm)
class TorchNNFunctionalBatchNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_norm_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.batch_norm(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7])
            return [result, input]
        input_tensor = torch.randn(20, 100, 35, 45)
        running_mean = torch.randn(100)
        running_var = torch.randn(100)
        weight = torch.randn(100)
        bias = torch.randn(100)
        training = True
        momentum = 0.1
        eps = 1e-5
        result = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)
        return [result, [input_tensor, running_mean, running_var, weight, bias, training, momentum, eps]]

