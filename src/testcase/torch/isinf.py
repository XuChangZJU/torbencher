import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.isinf)
class TorchIsinfTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isinf_correctness(self):
        # Step 1: Define random dimensions and size for tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Step 2: Create a random tensor and insert special values
        input_tensor = torch.randn(input_size)

        # Inserting inf, -inf, nan randomly
        num_of_elements = input_tensor.numel()
        indices_to_change = random.sample(range(num_of_elements), random.randint(1, num_of_elements))
        special_values = [float('inf'), -float('inf'), float('nan')]

        for idx in indices_to_change:
            value_to_insert = random.choice(special_values)
            input_tensor.view(-1)[idx] = value_to_insert

        # Step 3: Apply torch.isinf and return the result
        result = torch.isinf(input_tensor)
        return result
