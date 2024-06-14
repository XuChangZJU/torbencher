import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.normalize)
class TorchNnFunctionalNormalizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_normalize_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Random input tensor
        input_tensor = torch.randn(input_size)
        # Random p value
        p = random.uniform(1.0, 10.0)
        # Random dim value between 0 and dim-1
        dim = random.randint(0, dim - 1)
        # Calculate the result of normalize
        result = torch.nn.functional.normalize(input_tensor, p, dim)
        return result
    