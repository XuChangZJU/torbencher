import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.testing.make_tensor)
class TorchTestingMakeUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_make_tensor_correctness(self):
        # Randomly generate tensor shape
        dim = random.randint(1, 3)
        num_of_elements_each_dim = random.randint(1, 4)
        shape = [num_of_elements_each_dim for _ in range(dim)]

        # Randomly select dtype
        dtypes = [torch.float16,torch.float32, torch.float64]
        dtype = random.choice(dtypes)

        # Randomly select device
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        device = random.choice(devices)

        # Generate tensor using make_tensor
        result = torch.testing.make_tensor(shape, dtype=dtype, device=device)
        return result
