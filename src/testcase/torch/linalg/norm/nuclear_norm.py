
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.norm.nuclear_norm)
class TorchLinalgNormNuclearNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_nuclear_norm(self, input=None):
        if input is not None:
            result = torch.linalg.norm.nuclear_norm(input[0])
            return [result, input]
        a = torch.randn(3, 3)
        result = torch.linalg.norm.nuclear_norm(a)
        return [result, [a]]

