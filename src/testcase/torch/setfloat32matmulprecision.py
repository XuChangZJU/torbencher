import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.setfloat32matmulprecision)
class TorchSetfloat32matmulprecisionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_float32_matmul_precision_correctness(self):
    dim = random.randint(2, 3)  # Random dimension for the matrices
    num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for _ in range(dim)]
    
    A = torch.randn(input_size)  # Random matrix A
    B = torch.randn(input_size)  # Random matrix B

    # List of precision settings to test
    precisions = ["highest", "high", "medium"]

    results = []
    for precision in precisions:
        torch.set_float32_matmul_precision(precision)
        result = torch.matmul(A, B)
        results.append((precision, result))

    return results
