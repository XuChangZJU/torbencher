import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.batch_norm)
class TorchNnFunctionalBatchUnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_batch_norm_correctness(self):
        # Random input size
        dim = random.randint(2, 3)  # Dimension should be at least 2 for batch_norm
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Random parameters for batch_norm
        num_features = input_size[1]  # num_features should match the input tensor's channel dimension
        running_mean = torch.zeros(num_features)
        running_var = torch.ones(num_features)
        weight = torch.randn(num_features)
        bias = torch.randn(num_features)

        # Apply batch normalization
        result = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias)
        return result
