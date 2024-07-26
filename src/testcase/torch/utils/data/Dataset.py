import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_size):
        self.data_size = data_size
        self.data = torch.randn(self.data_size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data_size


@test_api(torch.utils.data.Dataset)
class TorchUtilsDataDatasetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dataset_correctness(self):
        data_size = random.randint(1, 100)  # Random data size for the dataset
        dataset = TestDataset(data_size)
        index = random.randint(0, data_size - 1)  # Random index within the dataset size
        result = dataset[index]
        return result
