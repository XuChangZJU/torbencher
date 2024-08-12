import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.profiler._KinetoProfile)
class TorchProfilerUkinetoprofileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_kineto_profile_correctness(self):
        # Randomly select activities from ProfilerActivity
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        # Randomly decide whether to record shapes
        record_shapes = random.choice([True, False])

        # Randomly decide whether to profile memory
        profile_memory = random.choice([True, False])

        # Randomly decide whether to record stack information
        with_stack = random.choice([True, False])

        # Randomly decide whether to estimate FLOPS
        with_flops = random.choice([True, False])

        # Randomly decide whether to record module hierarchy
        with_modules = random.choice([True, False])

        # Create a simple model for profiling
        model = torch.nn.Linear(10, 5)
        input_tensor = torch.randn(1, 10)

        # Start profiling
        with torch.profiler.profile(
                activities=activities,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules
        ) as prof:
            output = model(input_tensor)
