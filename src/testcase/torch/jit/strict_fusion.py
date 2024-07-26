import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.jit.strict_fusion)
class TorchJitStrictfusionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_strict_fusion_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size for the tensor
    
        tensor = torch.randn(input_size)  # Generate random tensor with the specified size
    
        @torch.jit.script
        def foo(x):
            return x + x + x  # Force fusion of additions
    
        result = foo(tensor)  # Call the scripted function with the random tensor
        return result
    