import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.ormqr)
class TorchTensorOrmqrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ormqr_correctness(self):
        # Randomly generate input parameters
        batch_size = random.randint(1, 3)
        m = random.randint(1, 5)
        n = random.randint(1, 5)
        k = random.randint(1, min(m, n))
        left = random.choice([True, False])
        transpose = random.choice([True, False])

        # Generate input tensors with random data
        input_size = [m if left else n, k]
        input_tensor = torch.randn([batch_size] + input_size)
        tau_size = [batch_size, min(input_size)]
        tau_tensor = torch.randn(tau_size)
        other_size = [batch_size, m, n]
        other_tensor = torch.randn(other_size)

        # Calculate the result
        result = input_tensor.ormqr(tau_tensor, other_tensor, left, transpose)
        return result
