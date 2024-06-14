import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.adjoint)
class TorchAdjointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adjoint_correctness(self):
        # Randomly generate the dimension of the tensor, at least 2 dimensions
        dim = random.randint(2, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size and data type as complex
        input_tensor = torch.randn(input_size, dtype=torch.complex64)
        # Calculate the adjoint of the input tensor using torch.adjoint
        result = torch.adjoint(input_tensor)
        # Return the result tensor
        return result
    