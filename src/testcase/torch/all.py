import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.all)
class TorchAllTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_all_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements per dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        tensor = torch.rand(input_size).bool()  # Random tensor with boolean values
        result = torch.all(tensor)
        return result
    