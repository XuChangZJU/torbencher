import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.schedule)
class TorchProfilerScheduleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_profiler_schedule_correctness(self):
        # Randomly generate parameters for the schedule
        skip_first = random.randint(0, 5)  # Random number of steps to skip initially
        wait = random.randint(1, 5)  # Random number of steps to wait
        warmup = random.randint(1, 5)  # Random number of steps for warmup
        active = random.randint(1, 5)  # Random number of steps for active recording
        repeat = random.randint(0, 5)  # Random number of cycles to repeat
    
        # Create the schedule callable
        schedule_callable = torch.profiler.schedule(
            skip_first=skip_first, 
            wait=wait, 
            warmup=warmup, 
            active=active, 
            repeat=repeat
        )
    
        # Generate a random step number to test the schedule
        step = random.randint(0, 50)
    
        # Get the phase for the given step
        phase = schedule_callable(step)
        return phase
    
    
    
    