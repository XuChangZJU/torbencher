import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.argmax)
class TorchTensorArgmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_argmax_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random tensor
        tensor = torch.randn(input_size)
        # Random dimension to calculate argmax
        dim_to_calculate = random.randint(-dim, dim - 1)  # dim_to_calculate should be in range [-dim, dim-1]
        # Calculate argmax
        result = tensor.argmax(dim=dim_to_calculate)
        return result
