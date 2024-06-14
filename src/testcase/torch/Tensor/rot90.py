import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.rot90)
class TorchTensorRot90TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rot90_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        input_tensor = torch.randn(input_size)
        # Generate random k
        k = random.randint(1, 4)
        # Generate random dims. The dims should contain two integers.
        dims = random.sample(range(0, dim), 2)
        result = input_tensor.rot90(k, dims)
        return result
    
    
    
    