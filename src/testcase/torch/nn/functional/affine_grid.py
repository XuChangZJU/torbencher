
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.affine_grid)
class TorchNNFunctionalAffineGridTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_affine_grid_common(self):
        
        a = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=torch.float)
        b = torch.Size([1, 1, 2, 2])
        result = torch.nn.functional.affine_grid(a, b)
        return result


