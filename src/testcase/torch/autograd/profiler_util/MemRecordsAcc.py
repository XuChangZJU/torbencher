import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.profiler_util.MemRecordsAcc)
class TorchAutogradProfilerutilMemrecordsaccTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mem_records_acc_correctness(self):
        # Randomly generate the number of memory records
        num_records = random.randint(1, 10)
        
        # Randomly generate memory records
        mem_records = []
        for _ in range(num_records):
            # Each memory record is a tuple of (address, size, allocation/deallocation)
            address = random.randint(1, 1000)
            size = random.randint(1, 1000)
            alloc_dealloc = random.choice([True, False])
            mem_records.append((address, size, alloc_dealloc))
        
        # Create MemRecordsAcc object
        mem_records_acc = torch.autograd.profiler.MemRecordsAcc(mem_records)
        
        # Accessing the memory records in the interval
        result = mem_records_acc.mem_records
        
        return result
    