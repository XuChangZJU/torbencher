import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.record_stream)
class TorchTensorRecordUstreamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skipIf(not torch.cuda.is_available(),"CUDA is not available")
    def test_record_stream_correctness(self):
        # Randomly select dimensions for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor
        tensor = torch.randn(input_size, device='cuda')

        # Create a CUDA stream
        stream = torch.cuda.Stream()

        # Record the stream with the tensor
        tensor.record_stream(stream)

        # Return the tensor to show the effect of record_stream
        return tensor
