import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.arcsin)
class TorchTensorArcsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arcsin_correctness(self):
    """
    Test the correctness of torch.Tensor.arcsin.
    """
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random tensor data within the range [-1, 1] to ensure valid arcsin operation.
    input_tensor = torch.rand(input_size) * 2 - 1 
    result = input_tensor.arcsin()
    return result
