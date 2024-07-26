import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.nonzero)
class TorchTensorNonzeroTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nonzero_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)
        # Get the indices of non-zero elements
        result = tensor.nonzero()
        return result
