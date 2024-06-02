
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.unsqueeze)
class TorchUnsqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unsqueeze(self, input=None):
        if input is not None:
            result = torch.unsqueeze(input[0], input[1])
            return [result, input]
        a = torch.tensor([1, 2, 3, 4])
        result = torch.unsqueeze(a, 1)
        return [result, [a, 1]]

