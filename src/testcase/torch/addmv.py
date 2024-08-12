import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.addmv)
class TorchAddmvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_addmv_correctness(self):
        dim_n = random.randint(1, 5)
        dim_m = random.randint(1, 5)
        input_size = [dim_n]
        mat_size = [dim_n, dim_m]
        vec_size = [dim_m]
        # Generate random tensors with valid sizes
        input_tensor = torch.randn(input_size)
        mat_tensor = torch.randn(mat_size)
        vec_tensor = torch.randn(vec_size)
        result = torch.addmv(input_tensor, mat_tensor, vec_tensor)
        return result
