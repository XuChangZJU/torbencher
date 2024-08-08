import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.independent.Independent)
class TorchDistributionsIndependentIndependentTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_Independent_correctness(self):
        # Randomly generate parameters for the base distribution
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        batch_shape = torch.Size([random.randint(1, 3) for _ in range(random.randint(1, 3))])
        event_shape = torch.Size([random.randint(1, 3) for _ in range(random.randint(1, 3))])

        # Create a base distribution (using Normal distribution as an example)
        loc = torch.randn(batch_shape + event_shape)
        scale = torch.randn(batch_shape + event_shape).abs()  # Scale should be positive
        base_distribution = torch.distributions.Normal(loc, scale)

        # Randomly select the number of batch dimensions to reinterpret
        reinterpreted_batch_ndims = random.randint(1, len(base_distribution.batch_shape))

        # Create an Independent distribution
        independent_distribution = torch.distributions.independent.Independent(base_distribution,
                                                                               reinterpreted_batch_ndims)

        # Sample from the Independent distribution
        sample = independent_distribution.sample()

        # Calculate the log probability of the sample
        log_prob = independent_distribution.log_prob(sample)

        return log_prob
