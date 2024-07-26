import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.jit.freeze)
class TorchJitFreezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_freeze_correctness(self):
        # Random dimensions for the weight matrix
        N = random.randint(1, 5)
        M = random.randint(1, 5)
        
        # Define a simple module with a parameter
        class MyModule(torch.nn.Module):
            def __init__(self, N, M):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(N, M))
                self.linear = torch.nn.Linear(M, M)  # Fix: Linear layer should have input size M
    
            def forward(self, input):
                output = self.weight.mm(input)
                output = self.linear(output)
                return output
    
        # Create a scripted module and freeze it
        scripted_module = torch.jit.script(MyModule(N, M).eval())
        frozen_module = torch.jit.freeze(scripted_module)
        
        # Check that parameters have been removed and inlined into the Graph as constants
        assert len(list(frozen_module.named_parameters())) == 0
        
        # Return the frozen module's code to show the effect of freezing
        return frozen_module.code
    def test_freeze_with_preserved_attrs(self):
        # Define a module with attributes to preserve
        class MyModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.modified_tensor = torch.tensor(10.)
                self.version = 1
    
            def forward(self, input):
                self.modified_tensor += 1
                return input + self.modified_tensor
    
        # Create a scripted module and freeze it with preserved attributes
        scripted_module = torch.jit.script(MyModule2().eval())
        frozen_module = torch.jit.freeze(scripted_module, ["version"])
        
        # Check that the preserved attribute still exists and can be modified
        assert frozen_module.version == 1
        frozen_module.version = 2
        
        # Check that the modified tensor is preserved and behaves correctly
        assert frozen_module(torch.tensor(1.)) == torch.tensor(12.)  # Fix: Ensure tensor type consistency
        assert frozen_module(torch.tensor(1.)) == torch.tensor(13.)
        
        # Return the frozen module's code to show the effect of freezing
        return frozen_module.code
    