import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.kumaraswamy.Kumaraswamy)
class TorchDistributionsKumaraswamyKumaraswamyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_kumaraswamy_sample_correctness(self):
        # Randomly generate concentration1 and concentration0
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        concentration1 = torch.rand(input_size) + 1e-5  # concentration1 > 0
        concentration0 = torch.rand(input_size) + 1e-5  # concentration0 > 0

        kumaraswamy_distribution = torch.distributions.kumaraswamy.Kumaraswamy(concentration1, concentration0)
        samples = kumaraswamy_distribution.sample()
        return samples
