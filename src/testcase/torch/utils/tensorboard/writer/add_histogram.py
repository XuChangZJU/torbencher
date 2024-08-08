import random

import torch
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_histogram)
class TorchUtilsTensorboardWriterAddUhistogramTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_add_histogram_correctness(self):
        writer = SummaryWriter()

        # Randomly generate the tag name
        tag = f"histogram_{random.randint(1, 100)}"

        # Random dimension for the tensor
        dim = random.randint(1, 4)

        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)

        # Generate random input size
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor data
        values = torch.randn(input_size)

        # Random step value
        step = random.randint(0, 100)

        # Add histogram to the writer
        writer.add_histogram(tag, values, step)

        # Close the writer
        writer.close()
