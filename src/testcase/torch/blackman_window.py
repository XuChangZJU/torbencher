
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.blackman_window)
class TorchBlackmanWindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_blackman_window(self, input=None):
        if input is not None:
            result = torch.blackman_window(input[0])
            return [result, input]
        a = 10
        result = torch.blackman_window(a)
        return [result, [a]]

