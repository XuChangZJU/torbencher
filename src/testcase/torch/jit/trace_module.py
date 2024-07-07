import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.trace_module)
class TorchJitTracemoduleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_trace_module_correctness(self):
        # Define the module with multiple methods
        class Net(torch.nn.Module):
            def __init__(self):
                super().
                self.conv = torch.nn.Conv2d(1, 1, 3)
    
            def forward(self, x):
                return self.conv(x)
    
            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight
    
        # Create an instance of the module
        n = Net()
    
        # Generate random input data for each method
        example_weight = torch.randn(1, 1, 3, 3)
        example_forward_input = torch.randn(1, 1, 3, 3)
    
        # Define the inputs dictionary for trace_module
        inputs = {'forward': example_forward_input, 'weighted_kernel_sum': example_weight}
    
        # Trace the module
        traced_module = torch.jit.trace_module(n, inputs)
    
        # Return the traced module
        return traced_module
    
    
    
    