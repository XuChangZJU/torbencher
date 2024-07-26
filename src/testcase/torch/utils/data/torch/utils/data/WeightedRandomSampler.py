import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.torch.utils.data.WeightedRandomSampler)
class TorchUtilsDataTorchUtilsDataWeightedrandomsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_WeightedRandomSampler_correctness(self):
        # Define parameters for WeightedRandomSampler
        num_samples = random.randint(1, 100)  # Number of samples to draw
        weights = torch.randn(random.randint(1, 100)).abs()  # Random weights (ensured to be non-negative)
        replacement = random.choice([True, False])  # Whether to sample with replacement
    
        # Create a WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)
    
        # Draw samples using the sampler
        samples = list(sampler)
    
        # Return the samples
        return samples
    