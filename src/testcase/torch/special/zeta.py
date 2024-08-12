import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.zeta)
class TorchSpecialZetaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_zeta_correctness(self):
        # Define random dimensions for the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random input tensors x and q
        x = torch.randn(input_size)
        x = torch.abs(x) + 1.1
        q = torch.randn(input_size)
        q = torch.abs(q) + 1

        # Calculate the Hurwitz zeta function
        result = torch.special.zeta(x, q)
        return result
