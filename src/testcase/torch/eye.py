
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.eye)
class TorchEyeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_eye(self, input=None):
        if input is not None:
            result = torch.eye(input[0], input[1])
            return result
        result = torch.eye(3, 4)
        return result

