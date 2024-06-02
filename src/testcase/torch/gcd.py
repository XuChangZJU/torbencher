
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.gcd)
class TorchGcdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gcd(self, input=None):
        if input is not None:
            result = torch.gcd(input[0], input[1])
            return [result, input]
        a = torch.randint(1, 10, (4,))
        b = torch.randint(1, 10, (4,))
        result = torch.gcd(a, b)
        return [result, [a, b]]


