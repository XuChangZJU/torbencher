import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.polygamma)
class TorchPolygammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_polygamma_correctness(self):
    # n: int, order of the polygamma function
    n = random.randint(1, 5)
    # Generate random dimension for the input tensor
    dim = random.randint(1, 4)
    # Generate random number of elements for each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Generate input tensor size
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # input: Tensor, the input tensor
    input = torch.randn(input_size)
    result = torch.polygamma(n, input)
    return result
