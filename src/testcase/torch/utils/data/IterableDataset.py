import torch
import random
import math

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.IterableDataset)
class TorchUtilsDataIterabledatasetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_iterable_dataset_correctness(self):
        # Randomly generate start and end for the dataset
        start = random.randint(0, 10)
        end = random.randint(start + 1, start + 10)  # Ensure end > start
    
        class MyIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, start, end):
                super(MyIterableDataset, self).__init__()
                self.start = start
                self.end = end
    
            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:  # single-process data loading
                    iter_start = self.start
                    iter_end = self.end
                else:  # in a worker process
                    per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
                    worker_id = worker_info.id
                    iter_start = self.start + worker_id * per_worker
                    iter_end = min(iter_start + per_worker, self.end)
                return iter(range(iter_start, iter_end))
    
        dataset = MyIterableDataset(start, end)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=random.randint(0, 4))
        result = list(dataloader)
        return result
    