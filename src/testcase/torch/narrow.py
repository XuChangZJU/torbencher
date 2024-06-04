
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.narrow)
class TorchNarrowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_narrow(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, ```python
       8, 9]])
        result = torch.narrow(a, 0, 0, 2)
        return result

