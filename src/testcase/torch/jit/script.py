import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.script)
class TorchJitScriptTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_jit_script_nn_Module(self):
        class MyModule(torch.nn.Module):
            def __init__(self, N, M):
                super().
                self.weight = torch.nn.Parameter(torch.rand(N, M))
                self.linear = torch.nn.Linear(N, M)
            def forward(self, input):
                output = self.weight.mv(input)
                output = self.linear(output)
                return output
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        N = random.randint(1, 10)
        M = random.randint(1, 10)
        module = MyModule(N, M)
        input = torch.randn(input_size)
        scripted_module = torch.jit.script(module)
        result = scripted_module(input)
        return result
    
    def test_torch_jit_script_standalone_function(self):
        @torch.jit.script
        def foo(x, y):
            if x.max() > y.max():
                r = x
            else:
                r = y
            return r
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size)
        y = torch.randn(input_size)
        result = foo(x, y)
        return result
    
    def test_torch_jit_script_dict(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        a = torch.randn(input_size)
        b = torch.randn(input_size)
        c = {
            'a': a,
            'b': b
        }
        result = torch.jit.script(c)
        return result
    
    def test_torch_jit_script_list(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        a = torch.randn(input_size)
        b = torch.randn(input_size)
        c = [a, b]
        result = torch.jit.script(c)
        return result
    
    # Automatically added function calls
    
    
    
    
    
    