import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.ldexp)
class TorchTensorLdexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ldexp_correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor1
        tensor1 = torch.randn(input_size)
        # Generate random tensor2, which has the same size as tensor1
        tensor2 = torch.randn(input_size)
        # Calculate ldexp result
        result = tensor1.ldexp(tensor2)
        return result
