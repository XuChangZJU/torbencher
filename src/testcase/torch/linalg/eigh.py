import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.eigh)
class TorchLinalgEighTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_eigh_correctness(self):
        # Randomly generate the input tensor A
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_size.extend([num_of_elements_each_dim, num_of_elements_each_dim])
        A = torch.randn(input_size)
        A = A + A.mT  # Make the matrix symmetric
        result = torch.linalg.eigh(A)
        return result
