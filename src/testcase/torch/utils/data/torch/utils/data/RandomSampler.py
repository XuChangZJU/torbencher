import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.torch.utils.data.RandomSampler)
class TorchUtilsDataTorchUtilsDataRandomsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_sampler_correctness(self):
        # Randomly generate parameters for RandomSampler
        num_samples = random.randint(1, 100)  # Random number of samples in the dataset
        replacement = random.choice([True, False])  # Randomly choose whether to sample with replacement
    
        # Create a RandomSampler instance
        random_sampler = torch.utils.data.RandomSampler(data_source=range(num_samples), replacement=replacement)
    
        # Get the sampled indices
        sampled_indices = list(random_sampler)
    
        return sampled_indices
    