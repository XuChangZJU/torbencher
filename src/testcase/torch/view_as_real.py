import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.view_as_real)
class TorchViewUasUrealTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_view_as_real_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list of input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random complex tensor
        input_tensor = torch.randn(input_size, dtype=torch.cfloat)
        # Apply view_as_real
        result = torch.view_as_real(input_tensor)
        return result
