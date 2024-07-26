import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.random_)
class TorchTensorRandomTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        tensor = torch.randn(input_size)
        from_value = random.randint(-10, 10) # Random from value
        to_value = random.randint(from_value + 1, from_value + 11) # Random to value, ensuring to > from
        result = tensor.random_(from_value, to_value)
        return result
    
    
    
    