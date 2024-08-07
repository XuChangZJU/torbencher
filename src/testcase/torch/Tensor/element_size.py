import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.element_size)
class TorchTensorElementUsizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_element_size_correctness(self):
        # Randomly choose a dtype from a list of common dtypes
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8]
        dtype = random.choice(dtypes)

        # Create a random tensor with the chosen dtype
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Use appropriate tensor creation function based on dtype
        if dtype in [torch.float32, torch.float64]:
            tensor = torch.randn(input_size, dtype=dtype)
        else:
            tensor = torch.randint(0, 10, input_size, dtype=dtype)

        element_size = tensor.element_size()
        return element_size
