import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.emit_nvtx)
class TorchAutogradProfilerEmitUnvtxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_emit_nvtx_correctness(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("Torch not compiled with CUDA enabled")

        # Randomly decide whether to enable the profiler
        enabled = random.choice([True, False])

        # Randomly decide whether to record shapes
        record_shapes = random.choice([True, False])

        # Random dimension for the tensor
        dim = random.randint(1, 4)

        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)

        # Generate random input size
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        tensor = torch.randn(input_size).cuda()  # Ensure tensor is on CUDA

        # Use the context manager to profile a simple operation
        with torch.autograd.profiler.emit_nvtx(enabled=enabled, record_shapes=record_shapes):
            result = tensor * 2  # Simple operation to show profiling effect

        return result
