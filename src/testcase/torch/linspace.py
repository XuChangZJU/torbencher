import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linspace)
class TorchLinspaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linspace_4d(self, input=None):
        if input is not None:
            result = torch.linspace(input[0], input[1], input[2])
            return [result, input]
        result = torch.linspace(0, 10, 5)
        return [result, [0, 10, 5]]

