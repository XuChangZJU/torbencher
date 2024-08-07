import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.graph.increment_version)
class TorchAutogradGraphIncrementUversionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_increment_version_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size, requires_grad=True)  # Create a tensor that requires grad
        original_version = tensor._version  # Get the original version of the tensor

        torch.autograd.graph.increment_version(tensor)  # Increment the version of the tensor
        updated_version = tensor._version  # Get the updated version of the tensor

        return updated_version != original_version  # Check if the version has been incremented
