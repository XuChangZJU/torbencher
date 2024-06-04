
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lcm)
class TorchLcmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lcm(self):
        
        a = torch.randint(1, 10, (4,))
        b = torch.randint(1, 10, (4,))
        result = torch.lcm(a, b)
        return result


