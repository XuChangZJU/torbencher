import torch
import random
from torch.utils.data import IterableDataset, ChainDataset

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


class RandomIterableDataset(IterableDataset):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        for _ in range(self.size):
            yield torch.randn(1).item()


@test_api(torch.utils.data.ChainDataset)
class TorchUtilsDataChaindatasetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chain_dataset_correctness(self):
        # Randomly generate the number of datasets to chain
        num_datasets = random.randint(2, 5)

        # Randomly generate the size of each dataset
        dataset_sizes = [random.randint(1, 10) for _ in range(num_datasets)]

        # Create the datasets
        datasets = [RandomIterableDataset(size) for size in dataset_sizes]

        # Chain the datasets
        chained_dataset = ChainDataset(datasets)

        # Collect all elements from the chained dataset
        result = list(chained_dataset)

        return result
