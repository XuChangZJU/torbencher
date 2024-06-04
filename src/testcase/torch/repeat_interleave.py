
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.repeat_interleave)
class TorchRepeatInterleaveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_repeat_interleave_tensor(self):
        
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = torch.repeat_interleave(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_repeat_interleave(self):
        
        a = torch.tensor([[1, 2], [3, 4]])
        result = torch.repeat_interleave(a, repeats=3, dim=1)
        return result

