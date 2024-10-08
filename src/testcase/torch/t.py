import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.t)
class TorchTTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_t_correctness(self):
        # For 0-D tensor
        input_tensor = torch.randn(())
        result = torch.t(input_tensor)
        return result
