import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sparsebsctensor)
class TorchSparsebsctensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sparse_bsc_tensor_correctness(self):
    # Random number of blocks
    ncolblocks = random.randint(1, 4)
    nrowblocks = random.randint(1, 4)
    blocksize = random.randint(1, 3)  # Random block size
    
    # Random values for ccol_indices, must have size ncolblocks + 1
    ccol_indices = torch.tensor([random.randint(0, nrowblocks) for _ in range(ncolblocks + 1)], dtype=torch.int64)
    ccol_indices[-1] = nrowblocks  # Ensure the last element is the total number of non-zeros
    
    # Random values for row_indices, must match the number of blocks defined by ccol_indices
    row_indices = torch.tensor([random.randint(0, nrowblocks - 1) for _ in range(ncolblocks)], dtype=torch.int64)
    
    # Random values, a (ncolblocks, blocksize, blocksize) tensor
    values = torch.randn((ncolblocks, blocksize, blocksize))
    
    result = torch.sparse_bsc_tensor(ccol_indices, row_indices, values)
    return result
