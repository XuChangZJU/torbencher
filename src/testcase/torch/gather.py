import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.gather)
class TorchGatherTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gather_4d(self, input=None):
        if input is not None:
            result = torch.gather(input[0], dim=0, index=input[1])
            return [result, input]
        a = torch.randn(4)
        indices = torch.tensor([0, 1, 2, 3])
        result = torch.gather(a, dim=0, index=indices)
        return [result, [a, indices]]

