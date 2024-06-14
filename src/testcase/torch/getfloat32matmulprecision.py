import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.getfloat32matmulprecision)
class TorchGetfloat32matmulprecisionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_float32_matmul_precision_correctness(self):
    # Retrieve current float32 matrix multiplication precision
    precision = torch.get_float32_matmul_precision()
    return precision
