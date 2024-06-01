import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.index_select)
class TorchIndexSelectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_select_4d(self, input=None):
        if input is not None:
            result = torch.index_select(input[0], dim=0, index=input[1])
            return [result, input]
        a = torch.randn(4)
        indices = torch.tensor([0, 2])
        result = torch.index_select(a, dim=0, index=indices)
        return [result, [a, indices]]

