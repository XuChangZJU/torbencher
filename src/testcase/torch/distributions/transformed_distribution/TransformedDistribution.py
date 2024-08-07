import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.transformed_distribution.TransformedDistribution)
class TorchDistributionsTransformedUdistributionTransformeddistributionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_TransformedDistribution_correctness(self):
        # Generate random parameters for the base distribution
        batch_shape = [random.randint(1, 5) for _ in range(random.randint(1, 3))]
        event_shape = [random.randint(1, 5) for _ in range(random.randint(1, 3))]

        # Create a base distribution (e.g., Normal)
        base_distribution = torch.distributions.Normal(torch.randn(batch_shape + event_shape),
                                                       torch.randn(batch_shape + event_shape).exp())

        # Create a list of transforms (e.g., ExpTransform, AffineTransform)
        transforms = [torch.distributions.transforms.ExpTransform(),
                      torch.distributions.transforms.AffineTransform(torch.randn(batch_shape + event_shape),
                                                                     torch.randn(batch_shape + event_shape).exp())]

        # Create a TransformedDistribution
        transformed_distribution = torch.distributions.transformed_distribution.TransformedDistribution(
            base_distribution, transforms)

        # Sample from the transformed distribution
        sample = transformed_distribution.sample()

        # Return the sample
        return sample
