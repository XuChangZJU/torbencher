import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.any)
class TorchTensorAnyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_any_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input_size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with values between 0 and 1
        tensor = torch.rand(input_size)
        # Apply any operation on the tensor
        result = tensor.any()
        return result
    
    
    