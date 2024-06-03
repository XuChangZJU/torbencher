
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.meshgrid)
class TorchMeshgridTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_meshgrid_4d(self, input=None):
        if input is not None:
            result = torch.meshgrid(input[0], input[1])
            return result
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = torch.meshgrid(a, b)
        return result

