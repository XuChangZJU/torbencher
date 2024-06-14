import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addmm)
class TorchAddmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmm_correctness(self):
    dim_n = random.randint(1, 4)
    dim_m = random.randint(1, 4)
    dim_p = random.randint(1, 4)
    input_size = [dim_n, dim_p]  
    mat1_size = [dim_n, dim_m]
    mat2_size = [dim_m, dim_p]

    input_tensor = torch.randn(input_size)
    mat1 = torch.randn(mat1_size)
    mat2 = torch.randn(mat2_size)
    result = torch.addmm(input_tensor, mat1, mat2)
    return result
