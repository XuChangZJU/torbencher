import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fx.GraphModule)
class TorchFxGraphmoduleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_GraphModule_correctness(self):
        # Generate random dimension and number of elements for the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random input tensor
        input_tensor = torch.randn(input_size)
    
        # Define a simple function to create a GraphModule from
        def func(x):
            return torch.add(x, 1)
    
        # Create a GraphModule from the function
        graph_module = torch.fx.symbolic_trace(func)
    
        # Run the GraphModule with the input tensor
        result = graph_module(input_tensor)
    
        return result
    
    
    
    