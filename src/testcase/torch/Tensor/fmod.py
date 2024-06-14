import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.fmod)
class TorchTensorFmodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fmod_correctness(self):
    # Generate random dimension and size for the input tensors
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random tensors with specified size
    input_tensor = torch.randn(input_size)
    divisor_tensor = torch.randn(input_size)  # Divisor tensor with the same size as input_tensor

    # Calculate the element-wise remainder of division
    result = input_tensor.fmod(divisor_tensor)

    return result
