import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.chunk)
class TorchTensorChunkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_chunk_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor
        tensor = torch.randn(input_size)

        # Randomly generate the number of chunks.
        # The number of chunks should be a divisor of the size of the dimension being chunked.
        chunks = random.randint(1, input_size[0])

        # Chunk the tensor
        result = tensor.chunk(chunks)
        return result
