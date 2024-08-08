import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.record_stream)
class TorchTensorRecordUstreamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_record_stream_correctness(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that PyTorch is installed with CUDA support.")

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
