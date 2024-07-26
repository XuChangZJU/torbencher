import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.unique)
class TorchTensorUniqueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unique_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor = torch.randint(0, 10, input_size)  # Random tensor with integer values between 0 and 9
        result = tensor.unique()
        return result
    
    
    
    