import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.alpha_dropout)
class TorchNnFunctionalAlphaUdropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_alpha_dropout_correctness(self):
        # Random input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Random p value (probability of an element to be dropped)
        p = random.uniform(0.1, 0.9)  # Should be between 0 and 1

        # Apply alpha dropout
        result = torch.nn.functional.alpha_dropout(input_tensor, p)
        return result
