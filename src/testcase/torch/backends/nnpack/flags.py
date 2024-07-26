import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.nnpack.flags)
class TorchBackendsNnpackFlagsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nnpack_flags_correctness(self):
        # Randomly choose a flag to test
        flag = random.choice(['enabled', 'disabled', 'auto'])

        # Set the flag
        if flag == 'enabled':
            torch.backends.mkldnn.enabled = True
        elif flag == 'disabled':
            torch.backends.mkldnn.enabled = False
        elif flag == 'auto':
            torch.backends.mkldnn.enabled = torch.backends.mkldnn.is_available()

        # Generate random tensor dimensions
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        tensor = torch.randn(input_size)

        # Perform a simple operation to see the effect of the flag
        result = torch.nn.functional.relu(tensor)

        return result
