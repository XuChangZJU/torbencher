import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.orgqr)
class TorchTensorOrgqrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_orgqr_correctness(self):
    # Randomly generate the dimensions for the input tensors
    m = random.randint(2, 5)  # Number of rows, must be at least 2
    n = random.randint(1, m)  # Number of columns, must be less than or equal to m
    k = random.randint(1, n)  # Number of elementary reflectors, must be less than or equal to n

    # Generate random input tensors
    input1 = torch.randn(m, n)
    input2 = torch.randn(k)

    # Perform the orgqr operation
    result = input1.orgqr(input2)
    return result
