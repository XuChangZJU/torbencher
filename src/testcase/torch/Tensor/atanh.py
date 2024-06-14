import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.atanh)
class TorchTensorAtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atanh_correctness(self):
    """
    Test the correctness of torch.Tensor.atanh with small scale random parameters.
    """
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    input_tensor = torch.randn(input_size)  # Generate random tensor data
    input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min()) * 2 - 1  # Normalize to (-1, 1) to make atanh valid
    result = input_tensor.atanh()
    return result
