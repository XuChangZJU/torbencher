
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.index_add)
class TorchIndexAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_add_1d(self):
        a = torch.ones(5, 3)
        dim = 0
        index = torch.tensor([0, 1, 2, 0])
        source = torch.arange(1, 13).reshape(4, 3)
        result = torch.index_add(a, dim=dim, index=index, source=source)
        return result

# torch.index_copy
