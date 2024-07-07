import torch
import random
import io


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.save)
class TorchJitSaveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_jit_save_correctness(self):
        # Define a simple module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).
    
            def forward(self, x):
                return x + random.randint(1, 10)  # Random addition to ensure variability
    
        # Create a ScriptModule from the defined module
        my_module = MyModule()
        script_module = torch.jit.script(my_module)
    
        # Save to a file
        file_name = 'scriptmodule_' + str(random.randint(1, 1000)) + '.pt'
        torch.jit.save(script_module, file_name)
    
        # Save to an io.BytesIO buffer
        buffer = io.BytesIO()
        torch.jit.save(script_module, buffer)
    
        # Save with extra files
        extra_files = {f'file_{random.randint(1, 100)}.txt': b'content'}
        torch.jit.save(script_module, file_name, _extra_files=extra_files)
    
        return file_name, buffer, extra_files
    
    
    
    