import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.diagonal)
class TorchDiagonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_diagonal_correctness(self):
        dim = random.randint(2, 4)  # dim >= 2
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
        offset = random.randint(-(num_of_elements_each_dim - 1),
                                num_of_elements_each_dim - 1)  # offset within the valid range
        dim1 = random.randint(0, dim - 1)  # dim1 within the valid dimension range
        dim2 = random.randint(0, dim - 1)  # dim2 within the valid dimension range
        while dim1 == dim2:  # Ensure dim1 and dim2 are different
            dim2 = random.randint(0, dim - 1)
        result = torch.diagonal(input_tensor, offset, dim1, dim2)
        return result
