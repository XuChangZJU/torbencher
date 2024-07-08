import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.profiler.profile)
class TorchProfilerProfileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_profiler_correctness(self):
        # Randomly choose activities
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        # Define a simple schedule function
        def schedule(step):
            if step == 0:
                return torch.profiler.ProfilerAction.WARMUP
            elif step == 1:
                return torch.profiler.ProfilerAction.RECORD
            else:
                return torch.profiler.ProfilerAction.NONE
        
        # Define a simple trace handler
        def trace_handler(prof):
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1)
            
        # Randomly generate tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Randomly generate tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        
        # Profile the addition of two tensors
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=trace_handler
        ) as p:
            result = torch.add(tensor1, tensor2)
            p.step()
        
        return result
    