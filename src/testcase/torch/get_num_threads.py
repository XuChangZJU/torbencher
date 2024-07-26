import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.get_num_threads)
class TorchGetnumthreadsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_num_threads_correctness(self):
        # Torch function to return the number of threads used for parallelizing CPU operations
        num_threads = torch.get_num_threads()
        return num_threads
    
    def test_set_num_threads_effect(self):
        # List of possible thread settings
        thread_settings = [1, 2, 4, 8, 16, 32]
    
        # Iterate over each setting and check if get_num_threads returns the set value
        results = []
        for threads in thread_settings:
            torch.set_num_threads(threads)  # Set the number of threads
            num_threads = torch.get_num_threads()  # Get the current number of threads
            results.append(num_threads == threads)  # Verify if it matches the set value
        
        return results
    
    
    
    
    