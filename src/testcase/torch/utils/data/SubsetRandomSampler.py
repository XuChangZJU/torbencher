import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.SubsetRandomSampler)
class TorchUtilsDataSubsetrandomsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_subset_random_sampler_correctness(self):
        # Randomly generate the length of indices list
        length_indices = random.randint(1, 100)
        # Generate a list of indices
        indices = list(range(length_indices))
        subset_random_sampler = torch.utils.data.SubsetRandomSampler(indices)
        return subset_random_sampler
    