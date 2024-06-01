import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.cross_entropy)
class TorchNNFunctionalCrossEntropyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross_entropy_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.cross_entropy(input[0], input[1], input[2], input[3], input[4], input[5])
            return [result, input]
        input_tensor = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        weight = torch.randn(5)
        ignore_index = -100
        reduction = 'mean'
        label_smoothing = 0.1
        result = torch.nn.functional.cross_entropy(input_tensor, target, weight, ignore_index, reduction, label_smoothing)
        return [result, [input_tensor, target, weight, ignore_index, reduction, label_smoothing]]

