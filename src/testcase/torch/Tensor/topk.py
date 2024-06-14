import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.topk)
class TorchTensorTopkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_topk_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensor 
        input_tensor = torch.randn(input_size)
        # Randomly generate k, k should be less than or equal to the size of the dimension specified by dim
        k = random.randint(1, input_size[dim - 1])
        
        # Call torch.Tensor.topk
        top_k_values, top_k_indices = input_tensor.topk(k)
        
        return top_k_values, top_k_indices
    