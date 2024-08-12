import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.dropout3d)
class TorchNnFunctionalDropout3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_dropout3d_correctness(self):
        # Random input size
        dim = random.randint(4, 5)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Random p value
        p = random.uniform(0.1, 0.9)  # Probability of an element to be zeroed

        # Apply dropout3d
        result = torch.nn.functional.dropout3d(input_tensor, p)

        return result
