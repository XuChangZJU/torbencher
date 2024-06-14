import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.mvlgamma)
class TorchTensorMvlgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mvlgamma__correctness(self):
    # Generate random dimension for the tensor
    dim = random.randint(1, 4)
    # Generate random number of elements each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Generate random input size
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Generate random tensor with values greater than (p - 1) / 2, where p is the input argument
    input_tensor = torch.randn(input_size) + torch.randint(1, 10, input_size) * 2
    # Apply mvlgamma_ operation in-place
    input_tensor.mvlgamma_(random.randint(2, 5))
    # Return the tensor after applying mvlgamma_
    return input_tensor
