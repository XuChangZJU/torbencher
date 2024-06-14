import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.isconj)
class TorchIsconjTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_conj_correctness(self):
    # Generate a random dimension for the tensor
    dim = random.randint(1, 4)  
    # Generate a random number of elements for each dimension
    num_of_elements_each_dim = random.randint(1, 5)  
    # Create a list representing the size of the tensor
    input_size = [num_of_elements_each_dim for i in range(dim)]  
    # Create a random tensor with the specified size
    input_tensor = torch.randn(input_size)
    # Set the conjugate bit of the tensor
    input_tensor.is_conj = bool(random.randint(0,1)) # Randomly set is_conj to True or False
    # Check if the tensor is conjugated
    result = torch.is_conj(input_tensor)
    return result
