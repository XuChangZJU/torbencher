
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.scatter)
class TorchScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_4d(self):
        a = torch.zeros(4)
        indices = torch.tensor([0, 1, 2, 3])
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.scatter(a, dim=0, index=indices, src=src)
        return result

