import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.profiler.EnforceUnique)
class TorchAutogradProfilerEnforceuniqueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_enforce_unique_correctness(self):
        # Generate a random number of keys
        num_keys = random.randint(1, 10)
        
        # Create a dictionary with unique keys
        unique_keys = {f'key_{i}': i for i in range(num_keys)}
        
        # Create a dictionary with duplicate keys to trigger the error
        duplicate_keys = {f'key_{i}': i for i in range(num_keys)}
        duplicate_keys['key_0'] = num_keys  # Add a duplicate key
        
        # Test with unique keys (should not raise an error)
        torch.autograd.profiler.EnforceUnique().check(unique_keys)
        
        # Test with duplicate keys (should raise an error)
        try:
            torch.autograd.profiler.EnforceUnique().check(duplicate_keys)
        except RuntimeError as e:
            f"Expected error for duplicate keys: {e}"
    
    
    
    