
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.split)
class TorchSplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_split_size(self):
        
        a = torch.randn(4, 4)
        result = torch.split(a, 2, dim=1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_split(self):
        
        a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = torch.split(a, [1, 2, 3, 4], dim=0)
        return result

