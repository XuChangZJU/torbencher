import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.addmm_)
class TorchTensorAddmmUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmm__correctness(self):
        # Random dimensions for the matrices
        m = random.randint(1, 4)
        n = random.randint(1, 4)
        p = random.randint(1, 4)

        # Random tensors for the operation
        mat1 = torch.randn(m, n)
        mat2 = torch.randn(n, p)
        input_tensor = torch.randn(m, p)

        # Perform the in-place addmm_ operation
        result = input_tensor.addmm_(mat1, mat2)
        return result
