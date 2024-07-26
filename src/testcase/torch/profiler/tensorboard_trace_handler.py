import torch
import random
import tempfile
import os

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.profiler.tensorboard_trace_handler)
class TorchProfilerTensorboardtracehandlerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensorboard_trace_handler_correctness(self):
        # Generate random parameters
        dir_name = tempfile.mkdtemp() # Randomly generated directory name
        worker_name = f'worker_{random.randint(0, 100)}' # Randomly generated worker name
    
        # Call the function
        result = torch.profiler.tensorboard_trace_handler(dir_name, worker_name)
    
        # Check if the directory was created
        assert os.path.exists(dir_name), f"Directory {dir_name} was not created."
    
        return result
    