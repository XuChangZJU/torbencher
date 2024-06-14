import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.matmul)
class TorchTensorMatmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_matmul_correctness(self):
    # Random dimensions for the tensors
    dim1 = random.randint(1, 4)
    dim2 = random.randint(1, 4)
    
    # Random number of elements for each dimension
    num_elements_dim1 = random.randint(1, 5)
    num_elements_dim2 = random.randint(1, 5)
    num_elements_dim3 = random.randint(1, 5)
    
    # Ensure valid matrix multiplication dimensions
    tensor1_size = [num_elements_dim1, num_elements_dim2]
    tensor2_size = [num_elements_dim2, num_elements_dim3]
    
    tensor1 = torch.randn(tensor1_size)
    tensor2 = torch.randn(tensor2_size)
    
    result = tensor1.matmul(tensor2)
    return result
