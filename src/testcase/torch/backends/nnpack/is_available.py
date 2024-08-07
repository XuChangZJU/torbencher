import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.nnpack.is_available)
class TorchBackendsNnpackIsUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nnpack_is_available(self):
        # Check if NNPACK is available
        nnpack_available = torch.backends.quantized.engine == 'qnnpack'

        # Generate random tensor size
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        tensor = torch.randn(input_size)

        # Perform an operation that would benefit from NNPACK if available
        result = torch.nn.functional.relu(tensor)

        return nnpack_available, result
