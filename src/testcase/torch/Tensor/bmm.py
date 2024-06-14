import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.bmm)
class TorchTensorBmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bmm_correctness(self):
    # Randomly generate the batch size, matrix dimensions for input tensors
    batch_size = random.randint(1, 10)
    dim1 = random.randint(1, 10)
    dim2 = random.randint(1, 10)
    dim3 = random.randint(1, 10)

    # Generate random tensors with specified dimensions
    input1 = torch.randn(batch_size, dim1, dim2)  # batch1
    input2 = torch.randn(batch_size, dim2, dim3)  # batch2
    
    # Perform batch matrix multiplication
    result = input1.bmm(input2)
    return result
