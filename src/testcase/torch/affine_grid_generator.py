
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.affine_grid_generator)
class TorchAffine_grid_generatorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_affine_grid_generator(self, input=None):
        if input is not None:
            result = torch.affine_grid_generator(input[0], input[1])
            return [result, input]
        theta = torch.rand(1, 2, 3)
        N = 1
        C = 3
        H = 5
        W = 5
        result = torch.affine_grid_generator(theta, torch.Size((N, C, H, W)))
        return [result, [theta, torch.Size((N, C, H, W))]]

