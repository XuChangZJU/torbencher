import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.eq)
class TorchTensorEqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_eq_correctness(self):
        # Generate random dimension and size for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors of the same size
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        # Compare the tensors element-wise for equality
        result = tensor1.eq(tensor2)
        return result
