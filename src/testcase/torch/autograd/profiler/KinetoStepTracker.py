import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.profiler.KinetoStepTracker)
class TorchAutogradProfilerKinetosteptrackerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kineto_step_tracker_correctness(self):
        # Create an instance of KinetoStepTracker
        kineto_step_tracker = torch.autograd.profiler.KinetoStepTracker()
    
        # Initialize step counts dictionary
        kineto_step_tracker.step_counts = {
            "ProfilerStep": 0,
            "Optimizer1Step": 0,
            "Optimizer2Step": 0
        }
    
        # Randomly generate step counts for different requesters
        profiler_step_count = random.randint(0, 100)
        optimizer1_step_count = random.randint(0, 100)
        optimizer2_step_count = random.randint(0, 100)
    
        # Set the step counts in the KinetoStepTracker
        kineto_step_tracker.step_counts["ProfilerStep"] = profiler_step_count
        kineto_step_tracker.step_counts["Optimizer1Step"] = optimizer1_step_count
        kineto_step_tracker.step_counts["Optimizer2Step"] = optimizer2_step_count
    
        # Calculate the expected global step count
        expected_global_step_count = max(profiler_step_count, optimizer1_step_count, optimizer2_step_count)
    
        # Increment the step count for a random requester
        requester = random.choice(["ProfilerStep", "Optimizer1Step", "Optimizer2Step"])
        kineto_step_tracker.step_counts[requester] += 1
    
        # Calculate the new expected global step count
        new_expected_global_step_count = max(kineto_step_tracker.step_counts.values())
    
        # Return the new global step count
        return new_expected_global_step_count
    
    
    
    