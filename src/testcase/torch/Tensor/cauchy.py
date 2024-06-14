import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cauchy)
class TorchTensorCauchyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cauchy__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        median = random.uniform(-10.0, 10.0)  # Random median value between -10.0 and 10.0
        sigma = random.uniform(0.1, 10.0)  # Random sigma value between 0.1 and 10.0 (sigma should be positive)
        result = input_tensor.cauchy_(median, sigma)
        return result
    