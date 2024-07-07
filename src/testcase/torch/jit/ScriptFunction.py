import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.ScriptFunction)
class TorchJitScriptfunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def my_function(x, y):
        return x + y
    
    def test_script_function_correctness(self):
        # Script the function
        scripted_function = torch.jit.script(my_function)
    
        # Generate random dimensions for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Call the scripted function
        result = scripted_function(tensor1, tensor2)
        return result
    
    
    
    