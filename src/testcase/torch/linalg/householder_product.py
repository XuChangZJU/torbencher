import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.householder_product)
class TorchLinalgHouseholderUproductTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_householder_product_correctness(self):
        # Randomly generate the size of matrix A
        m = random.randint(2, 5)  # m should be greater than or equal to n
        n = random.randint(1, m)
        # k should be less than or equal to n
        k = random.randint(1, n)
        # Generate random input tensors A and tau
        A = torch.randn(m, n)
        tau = torch.randn(k)
        # Calculate the householder product
        result = torch.linalg.householder_product(A, tau)
        return result
