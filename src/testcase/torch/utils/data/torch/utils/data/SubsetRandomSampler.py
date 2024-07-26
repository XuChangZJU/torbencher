import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.torch.utils.data.SubsetRandomSampler)
class TorchUtilsDataTorchUtilsDataSubsetrandomsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_SubsetRandomSampler_correctness(self):
        # Create a list of indices
        num_of_indices = random.randint(1, 100)  # Random number of indices
        indices = list(range(num_of_indices))

        # Create a SubsetRandomSampler
        subset_sampler = torch.utils.data.SubsetRandomSampler(indices)

        # Get a batch of indices from the sampler
        batch_size = random.randint(1, num_of_indices)  # Random batch size
        sampled_indices = subset_sampler.__iter__().__next__()

        return sampled_indices
