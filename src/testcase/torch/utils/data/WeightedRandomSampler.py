import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.WeightedRandomSampler)
class TorchUtilsDataWeightedrandomsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_WeightedRandomSampler_correctness(self):
        # Randomly generate weights
        weights_length = random.randint(1, 10)
        weights = [random.uniform(0.1, 1.0) for _ in range(weights_length)]
        # Randomly generate num_samples, ensuring it's less than or equal to the length of weights for replacement=False
        num_samples = random.randint(1, weights_length)
        # Randomly choose replacement as True or False
        replacement = random.choice([True, False])
        
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)
        samples = list(sampler)
        return samples
    