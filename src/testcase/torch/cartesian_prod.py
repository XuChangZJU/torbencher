
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cartesian_prod)
class TorchCartesianProdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cartesian_prod_4d(self, input=None):
        if input is not None:
            result = torch.cartesian_prod(*input)
            return [result, input]
        a = torch.tensor([1, 2])
        b = torch.tensor([3, 4])
        result = torch.cartesian_prod(a, b)
        return [result, [a, b]]

