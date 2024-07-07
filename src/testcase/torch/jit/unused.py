import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.unused)
class TorchJitUnusedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_jit_unused_correctness(self):
        class MyModule(torch.nn.Module):
            def __init__(self, use_memory_efficient):
                super().
                self.use_memory_efficient = use_memory_efficient
    
            @torch.jit.unused
            def memory_efficient(self, x):
                return x + 10
    
            def forward(self, x):
                if self.use_memory_efficient:
                    return self.memory_efficient(x)
                else:
                    return x + 10
    
        # Randomly decide whether to use memory efficient mode
        use_memory_efficient = random.choice([True, False])
        module = MyModule(use_memory_efficient)
        scripted_module = torch.jit.script(module)
    
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Return the result of the forward pass
        return scripted_module(input_tensor)
    
    
    
    