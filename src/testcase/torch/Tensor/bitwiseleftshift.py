import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.bitwiseleftshift)
class TorchTensorBitwiseleftshiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_left_shift__correctness(self):
    # Define the dimension and size of the tensors randomly.
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Create random tensors with integer values. The values should be non-negative to avoid unintended behavior of bitwise shift with negative numbers.
    input_tensor = torch.randint(0, 10, input_size)
    shift_amount = torch.randint(0, 10, input_size) # Shift amount should be non-negative.

    # Perform in-place bitwise left shift operation.
    input_tensor.bitwise_left_shift_(shift_amount)

    return input_tensor 
