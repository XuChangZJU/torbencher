import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.softsign)
class TorchNnFunctionalSoftsignTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_softsign_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random input tensor
        input_tensor = torch.randn(input_size)
        # Calculate softsign
        result = torch.nn.functional.softsign(input_tensor)
        return result
