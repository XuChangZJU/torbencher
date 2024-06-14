import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.lu)
class TorchTensorLuTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_correctness(self):
    # Generate random dimension for the tensor
    dim = random.randint(2, 4)
    # Generate random number of elements each dimension
    num_of_elements_each_dim = random.randint(2, 5)
    # Generate input_size
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Generate random tensor
    input_tensor = torch.randn(input_size)
    # Perform LU decomposition
    result = input_tensor.lu()
    # Return the result
    return result
