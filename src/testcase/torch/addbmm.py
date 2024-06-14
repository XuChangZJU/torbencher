import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addbmm)
class TorchAddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addbmm_correctness(self):
    # Define the batch size
    batch_size = random.randint(1, 10)
    # Define the dimensions for matrices
    n = random.randint(1, 10)  
    m = random.randint(1, 10) 
    p = random.randint(1, 10) 
    # Generate random tensors
    input_tensor = torch.randn(n, p)
    batch1 = torch.randn(batch_size, n, m)
    batch2 = torch.randn(batch_size, m, p)
    result = torch.addbmm(input_tensor, batch1, batch2)
    return result
