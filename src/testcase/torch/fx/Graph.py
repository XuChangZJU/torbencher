import torch
import torch.fx
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fx.Graph)
class TorchFxGraphTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fx_graph_correctness(self):
        class RandomModule(torch.nn.Module):
            def __init__(self):
                super().
                input_dim = random.randint(1, 5)
                output_dim = random.randint(1, 5)
                self.linear = torch.nn.Linear(input_dim, output_dim)
    
            def forward(self, x):
                return torch.relu(self.linear(x))
    
        # Create a random module instance
        module = RandomModule()
        
        # Generate random input tensor
        input_dim = module.linear.in_features
        batch_size = random.randint(1, 5)
        input_tensor = torch.randn(batch_size, input_dim)
        
        # Trace the module to create a Graph
        traced_module = torch.fx.symbolic_trace(module)
        
        # Print the graph to verify correctness
        graph = traced_module.graph
        graph
        
        return graph
    