import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.binary_cross_entropy)
class TorchNNFunctionalBinaryCrossEntropyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.binary_cross_entropy(input[0], input[1], input[2], input[3])
            return [result, input]
        input_tensor = torch.randn(3, requires_grad=True)
        target = torch.empty(3).random_(2)
        weight = torch.randn(3)
        reduction = 'mean'
        result = torch.nn.functional.binary_cross_entropy(input_tensor, target, weight, reduction)
        return [result, [input_tensor, target, weight, reduction]]

