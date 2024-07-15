import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.jit.export)
class TorchJitExportTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_jit_export_correctness(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
    
            def implicitly_compiled_method(self, x):
                return x + 99
    
            def forward(self, x):
                return x + 10
    
            @torch.jit.export
            def another_forward(self, x):
                return self.implicitly_compiled_method(x)
    
            def unused_method(self, x):
                return x - 20
    
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random tensor input
        input_tensor = torch.randn(input_size)
    
        # Create an instance of MyModule and script it
        scripted_module = torch.jit.script(MyModule())
    
        # Test the scripted methods
        result_forward = scripted_module(input_tensor)
        result_another_forward = scripted_module.another_forward(input_tensor)
    
        return result_forward, result_another_forward
    