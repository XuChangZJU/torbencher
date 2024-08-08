import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.to)
class TorchTensorToTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_to_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)

        # Randomly choose a dtype
        dtype = random.choice([torch.float32, torch.float64, torch.int32, torch.int64])

        # Randomly choose a device
        device = random.choice(['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu'])

        # Convert tensor to the chosen dtype and device
        result = tensor.to(device=device, dtype=dtype)

        return result
