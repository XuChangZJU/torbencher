import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.ScriptModule)
class TorchJitScriptmoduleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_script_module_correctness(self):
        # Define a simple ScriptModule with a forward method
        class MyScriptModule(torch.nn.Module):
            def __init__(self):
                super(MyScriptModule, self).
                self.linear = torch.nn.Linear(10, 5)
    
            def forward(self, x):
                return self.linear(x)
    
        # Create an instance of the ScriptModule
        script_module = MyScriptModule()
    
        # Convert the module to a ScriptModule
        script_module = torch.jit.script(script_module)
    
        # Generate random input tensor with appropriate size
        input_tensor = torch.randn(1, 10)  # Batch size of 1, input features of 10
    
        # Run the forward method of the ScriptModule
        result = script_module(input_tensor)
        return result
    
    
    
    