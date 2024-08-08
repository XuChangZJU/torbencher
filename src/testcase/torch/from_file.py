import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.from_file)
class TorchFromUfileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_from_file_correctness(self):
        # Randomly generate parameters for torch.from_file
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor = torch.randn(input_size)
        filename = "test_from_file.pt"
        tensor.numpy().tofile(filename)
        shared = random.choice([True, False])  # Randomly choose whether to share memory
        size = torch.numel(tensor)  # Size should match the number of elements in the tensor

        # Call torch.from_file with the generated parameters
        result = torch.from_file(filename, shared, size)

        return result
