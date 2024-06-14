import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.asinh)
class TorchTensorAsinhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_asinh_correctness(self):
    """
    Test the correctness of torch.Tensor.asinh with small scale random parameters.
    """
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for i in range(dim)]

    input_tensor = torch.randn(input_size)  # Random tensor
    result = input_tensor.asinh()
    return result
