import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.truedivide)
class TorchTruedivideTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_true_divide_correctness(self):
    # Generate random dimension and size for the tensors
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random tensors
    dividend = torch.randn(input_size)  
    divisor = torch.randn(input_size) # divisor should not be zero. torch.randn generate random numbers from a standard normal distribution, so it's extremely unlikely to have a zero.

    # Perform element-wise division
    result = torch.true_divide(dividend, divisor)
    return result
