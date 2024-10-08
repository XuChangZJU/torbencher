import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.Dropout3d)
class TorchNnDropout3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip("内部随机")
    def test_dropout3d_correctness(self):
        # Random input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Random p value
        p = random.uniform(0.1, 0.9)  # probability of an element to be zeroed

        # Create Dropout3d module
        dropout3d = torch.nn.Dropout3d(p)

        # Apply dropout
        result = dropout3d(input_tensor)
        return result
