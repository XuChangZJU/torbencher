import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.logsumexp)
class TorchTensorLogsumexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logsumexp_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size list
    
        tensor = torch.randn(input_size)  # Generate random tensor with the specified size
        dim_to_reduce = random.randint(0, dim - 1)  # Randomly choose a dimension to reduce
    
        result = tensor.logsumexp(dim_to_reduce)  # Apply logsumexp on the chosen dimension
        return result
    
    
    
    