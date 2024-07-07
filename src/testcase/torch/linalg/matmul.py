import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matmul)
class TorchLinalgMatmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_matmul_correctness(self):
        # Randomly choose dimensions for the tensors
        dim1 = random.randint(1, 4)
        dim2 = random.randint(1, 4)
        common_dim = random.randint(1, 5)  # Common dimension for matrix multiplication
    
        # Generate random sizes for the tensors
        input_size1 = [random.randint(1, 5) for _ in range(dim1)]
        input_size2 = [random.randint(1, 5) for _ in range(dim2)]
    
        # Ensure the inner dimensions match for valid matrix multiplication
        if len(input_size1) == 1:
            input_size1 = [common_dim]
        else:
            input_size1[-1] = common_dim
    
        if len(input_size2) == 1:
            input_size2 = [common_dim]
        else:
            input_size2[0] = common_dim
    
        # Ensure the outer dimensions match for valid matrix multiplication
        if len(input_size1) > 1 and len(input_size2) > 1:
            input_size2[1] = input_size1[0]
    
        # Generate random tensors
        tensor1 = torch.randn(input_size1)
        tensor2 = torch.randn(input_size2)
    
        # Perform matrix multiplication
        result = torch.matmul(tensor1, tensor2)
        return result
    
    
    
    