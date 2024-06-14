import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.asin)
class TorchTensorAsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_asin__correctness(self):
    """
    Test the correctness of the torch.Tensor.asin_() operator.
    """
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    input_tensor = torch.randn(input_size) # Generate random tensor data
    input_tensor.requires_grad = True # Enable gradient calculation
    input_tensor.asin_() # In-place asin operation
    return input_tensor
