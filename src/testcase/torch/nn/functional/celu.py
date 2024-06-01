import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.celu)
class TorchNNFunctionalCELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_celu_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.celu(input[0], input[1])
            return [result, input]
        input_tensor = torch.randn(3, 4, 5)
        alpha = 1.0
        result = torch.nn.functional.celu(input_tensor, alpha)
        return [result, [input_tensor, alpha]]

