
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.geqrf)
class TorchGeqrfTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_geqrf_2d(self):
        a = torch.randn(3, 4)
        result = torch.geqrf(a)
        return result

# torch.ger
