import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.meshgrid)
class TorchMeshgridTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_meshgrid_correctness(self):
        # Define the dimensions and number of elements for the input tensors
        dim = random.randint(1, 3)  # Random dimension for the tensors (up to 3 for visualization)
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_sizes = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensors based on the input sizes
        tensors = [torch.randn(input_sizes[i]) for i in range(dim)]

        # Apply torch.meshgrid
        result = torch.meshgrid(*tensors)
        return result
