import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.nanmean)
class TorchTensorNanmeanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nanmean_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5)
        # Random input size
        input_size=[num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor with nan
        tensor = torch.randn(input_size)
        tensor[tensor > 0.5] = float('nan')
        # Randomly choose a dimension to calculate the mean
        dim = random.randint(0, dim - 1)
        result = tensor.nanmean(dim)
        return result
    
    
    
    