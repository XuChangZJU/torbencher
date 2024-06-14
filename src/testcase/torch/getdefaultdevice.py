import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.getdefaultdevice)
class TorchGetdefaultdeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_default_dtype_correctness(self):
    initial_dtype = torch.get_default_dtype()  # Save the initial default dtype
    
    # Generate random dtype selection to change the default dtype
    dtype_choices = [torch.float32, torch.float64, torch.float16]
    random_dtype = random.choice(dtype_choices)  # Randomly select a new default dtype
    torch.set_default_dtype(random_dtype)
    
    # Check if default dtype has been set correctly
    result = torch.get_default_dtype()
    
    # Reset to initial default dtype for stability in subsequent tests
    torch.set_default_dtype(initial_dtype)
    
    return result
