import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.affine_grid)
class TorchNNFunctionalAffineGridTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_affine_grid_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.affine_grid(input[0], input[1])
            return [result, input]
        theta = torch.randn(2, 3, 3)
        size = torch.Size([2, 3, 24, 24])
        result = torch.nn.functional.affine_grid(theta, size)
        return [result, [theta, size]]

