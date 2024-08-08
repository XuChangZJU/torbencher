import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.gammaln)
class TorchSpecialGammalnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_gammaln_correctness(self):
        # Generate random input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)

        # Calculate gammaln
        result = torch.special.gammaln(input_tensor)

        return result
