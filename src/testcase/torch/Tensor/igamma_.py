import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.igamma_)
class TorchTensorIgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_igamma__correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor1
        tensor1 = torch.randn(input_size)
        # Generate random tensor2, make sure all the elements are greater than 0
        tensor2 = torch.rand(input_size) + 1e-5
        # Calculate the result using igamma_
        result = tensor1.igamma_(tensor2)
        # Return the result
        return result
