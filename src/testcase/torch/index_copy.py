
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.index_copy)
class TorchIndexCopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_copy_1d(self, input=None):
        if input is not None:
            result = torch.index_copy(input[0], dim=input[1], index=input[2], source=input[3])
            return [result, input]
        a = torch.zeros(5, 3)
        dim = 0
        index = torch.tensor([0, 1, 2, 0])
        source = torch.arange(1, 13).reshape(4, 3)
        result = torch.index_copy(a, dim=dim, index=index, source=source)
        return [result, [a, dim, index, source]]

# torch.index_fill
