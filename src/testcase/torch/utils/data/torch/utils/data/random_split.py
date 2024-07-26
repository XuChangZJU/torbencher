import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.torch.utils.data.random_split)
class TorchUtilsDataTorchUtilsDataRandomsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_split_correctness(self):
        # Randomly generate the length of the dataset
        dataset_length = random.randint(1, 100)
        # Randomly generate the lengths of the splits
        split_lengths = [random.randint(1, dataset_length) for _ in range(random.randint(1, 5))]
        # Ensure the sum of split lengths does not exceed the dataset length
        split_lengths[-1] = dataset_length - sum(split_lengths[:-1])
        # Generate a random dataset
        dataset = torch.randn(dataset_length)
        # Apply random_split
        result = torch.utils.data.random_split(dataset, split_lengths)
        return result
    