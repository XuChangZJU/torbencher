import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.set_module)
class TorchUtilsSetmoduleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_module_correctness(self):
        class DummyClass:
            pass
    
        # Create a dummy class instance
        dummy_instance = DummyClass()
    
        # Randomly generate a module name
        module_name = f"module_{random.randint(1, 100)}"
    
        # Set the module attribute using torch.utils.set_module
        torch.utils.set_module(dummy_instance, module_name)
    
        # Check if the module attribute is set correctly
        result = dummy_instance.__module__
        return result
    
    
    
    