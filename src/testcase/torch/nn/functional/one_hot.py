
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.one_hot)
class TorchNNFunctionalOneHotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_one_hot_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.one_hot(input[0], num_classes=input[1])
            return [result, input]
        a = torch.arange(0, 5) % 3
        b = 5
        result = torch.nn.functional.one_hot(a, num_classes=b)
        return [result, [a, b]]


