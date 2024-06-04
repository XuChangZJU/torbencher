
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lu_unpack)
class TorchLuUnpackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_unpack_4d(self):
        a = torch.randn(4, 4)
        lu_data, pivots = torch.lu(a)
        result = torch.lu_unpack(lu_data, pivots)
        return result

