import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.addbmm)
class TorchTensorAddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addbmm_correctness(self):
        # Random dimensions for the batch matrices
        batch_size = random.randint(1, 4)
        M = random.randint(1, 5)
        N = random.randint(1, 5)
        P = random.randint(1, 5)
        
        # Random tensors for batch1 and batch2
        batch1 = torch.randn(batch_size, M, N)
        batch2 = torch.randn(batch_size, N, P)
        
        # Random tensor for the input tensor
        input_tensor = torch.randn(M, P)
        
        # Perform the addbmm operation
        result = input_tensor.addbmm(batch1, batch2)
        
        return result
    
    
    
    