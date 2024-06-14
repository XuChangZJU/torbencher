import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.greater)
class TorchTensorGreaterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_greater__correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    input = torch.randn(input_size)
    other = torch.randn(input_size) # other should have same size as input
    input.greater_(other)
    return input
