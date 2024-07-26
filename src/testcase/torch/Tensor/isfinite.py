import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.isfinite)
class TorchTensorIsfiniteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isfinite_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)

        # Introduce some infinities and NaNs
        num_infs = random.randint(0, num_of_elements_each_dim)  # Random number of infinities
        num_nans = random.randint(0, num_of_elements_each_dim)  # Random number of NaNs

        for _ in range(num_infs):
            idx = tuple(random.randint(0, num_of_elements_each_dim - 1) for _ in range(dim))
            tensor[idx] = float('inf') if random.random() > 0.5 else float('-inf')

        for _ in range(num_nans):
            idx = tuple(random.randint(0, num_of_elements_each_dim - 1) for _ in range(dim))
            tensor[idx] = float('nan')

        result = tensor.isfinite()
        return result
