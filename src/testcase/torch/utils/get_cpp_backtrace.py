import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.get_cpp_backtrace)
class TorchUtilsGetcppbacktraceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_cpp_backtrace_correctness(self):
        frames_to_skip = random.randint(0, 10)  # Random number of frames to skip
        maximum_number_of_frames = random.randint(1, 100)  # Random maximum number of frames to return
        result = torch.utils.get_cpp_backtrace(frames_to_skip, maximum_number_of_frames)
        return result
    
    
    
    