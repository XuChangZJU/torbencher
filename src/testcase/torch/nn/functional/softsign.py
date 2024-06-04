
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.softsign)
class TorchNNFunctionalSoftsignTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softsign_2d(self):
        
        a = torch.randn(3, 2)
        result = torch.nn.functional.softsign(a)
        return result


