import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.export.ModuleCallEntry)
class TorchExportModulecallentryTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ModuleCallEntry_correctness(self):
        # Define parameters for ModuleCallEntry
        fqn = "my_module.MyClass.forward"  # Randomly generated valid fqn
        signature = None  # Use default value
    
        # Create a ModuleCallEntry object
        module_call_entry = torch.export.ModuleCallEntry(fqn, signature)
    
        # Return the ModuleCallEntry object for inspection
        return module_call_entry
    