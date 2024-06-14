import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.ceil)
class TorchTensorCeilTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ceil__correctness(self):
    # Generate random dimension for the tensor
    dim = random.randint(1, 4)  
    # Generate random number of elements each dimension
    num_of_elements_each_dim = random.randint(1,5) 
    # Generate input_size
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    # Generate random tensor 
    input_tensor = torch.randn(input_size)
    # Perform ceil_ operation in-place
    input_tensor.ceil_()
    # Return the tensor after ceil_ operation
    return input_tensor
