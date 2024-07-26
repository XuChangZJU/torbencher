import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.jit_python_reference)
class TorchJitJitpythonreferenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_jit_python_reference_correctness(self):
        # Randomly generate the size of the tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)

        # Apply torch.jit.script to the tensor
        @torch.jit.script
        def process_tensor(tensor):
            return tensor

        result = process_tensor(tensor)
        return result
