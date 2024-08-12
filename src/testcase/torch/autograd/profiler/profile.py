import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.profile)
class TorchAutogradProfilerProfileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_profiler_correctness(self):
        # Randomly decide whether to enable the profiler
        enabled = random.choice([True, False])

        # Randomly decide whether to use CUDA
        use_cuda = random.choice([True, False])

        # Randomly decide whether to record shapes
        record_shapes = random.choice([True, False])

        # Randomly decide whether to estimate FLOPs
        with_flops = random.choice([True, False])

        # Randomly decide whether to profile memory
        profile_memory = random.choice([True, False])

        # Randomly decide whether to record source information
        with_stack = random.choice([True, False])

        # Randomly decide whether to record module hierarchy
        with_modules = random.choice([True, False])

        # Randomly decide whether to use Kineto profiler
        use_kineto = random.choice([True, False])

        # Randomly decide whether to profile CPU events
        use_cpu = random.choice([True, False])

        # Random dimension for the tensor
        dim = random.randint(1, 4)

        # Random number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)

        # Generate random input size
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        tensor = torch.randn(input_size, requires_grad=True)

        # Use the profiler context manager
        with torch.autograd.profiler.profile(
                enabled=enabled,
                use_cuda=use_cuda,
                record_shapes=record_shapes,
                with_flops=with_flops,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_modules=with_modules,
                use_kineto=use_kineto,
                use_cpu=use_cpu
        ) as prof:
            # Perform some operations
            y = tensor ** 2
            y.backward()

        # Return the profiler key averages table
        return prof.key_averages().table(sort_by="self_cpu_time_total")
