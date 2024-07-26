import torch
import random
from torch.utils.data import DataLoader, Dataset, get_worker_info

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


class RandomDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@test_api(torch.utils.data.get_worker_info)
class TorchUtilsDataGetworkerinfoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        return worker_info

    def test_get_worker_info_correctness(self):
        dataset_size = random.randint(10, 100)  # Random dataset size between 10 and 100
        batch_size = random.randint(1, 10)  # Random batch size between 1 and 10
        num_workers = random.randint(1, 4)  # Random number of workers between 1 and 4

        dataset = RandomDataset(dataset_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)

        for batch in dataloader:
            worker_info = get_worker_info()
            if worker_info is not None:
                f"Worker ID: {worker_info.id}, Num Workers: {worker_info.num_workers}, Seed: {worker_info.seed}"
                f"Dataset in Worker: {worker_info.dataset}"
