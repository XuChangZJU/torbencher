import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.inner)
class TorchTensorInnerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inner_correctness(self):
        dim1 = random.randint(1, 4)  # Random dimension for the first tensor
        dim2 = random.randint(1, 4)  # Random dimension for the second tensor
        num_of_elements_each_dim1 = random.randint(1, 5)  # Random number of elements each dimension for the first tensor
        num_of_elements_each_dim2 = random.randint(1, 5)  # Random number of elements each dimension for the second tensor
    
        # Ensure the last dimension of tensor1 matches the first dimension of tensor2
        tensor1_size = [num_of_elements_each_dim1 for _ in range(dim1)]
        tensor2_size = [num_of_elements_each_dim2 for _ in range(dim2)]
        
        if dim1 > 1 and dim2 > 1:
            tensor1_size[-1] = tensor2_size[0]
    
        tensor1 = torch.randn(tensor1_size)
        tensor2 = torch.randn(tensor2_size)
    
        result = torch.inner(tensor1, tensor2)
        return result
    
    
    
    