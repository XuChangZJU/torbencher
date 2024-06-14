import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.isfloatingpoint)
class TorchTensorIsfloatingpointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_floating_point_correctness(self):
    # Randomly generate dimension and size for the tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Create a floating-point tensor
    floating_point_tensor = torch.randn(input_size) 

    # Create an integer tensor
    integer_tensor = torch.randint(0, 10, input_size) 

    # Check if the tensors are correctly identified as floating-point or not
    result = floating_point_tensor.is_floating_point() and not integer_tensor.is_floating_point()
    return result
