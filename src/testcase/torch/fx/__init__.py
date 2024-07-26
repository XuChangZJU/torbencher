import torch
import random
import torch.fx

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fx.__init__)
class TorchFxInitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fx_init_correctness(self):
        # Randomly generate a name for the GraphModule
        module_name = f"module_{random.randint(1, 1000)}"
        
        # Create a random tensor to serve as an example input
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        example_input = torch.randn(input_size)
        
        # Create a simple function to trace
        def simple_function(x):
            return x + 1
        
        # Create a symbolic trace of the function
        traced = torch.fx.symbolic_trace(simple_function)
        
        # Initialize the GraphModule
        graph_module = torch.fx.GraphModule(traced, traced.graph, module_name)
        
        # Return the graph module and example input for inspection
        return graph_module, example_input
    from .Interpreter import TorchFxInterpreterTestCase
from .Transformer import TorchFxTransformerTestCase
from .GraphModule import TorchFxGraphmoduleTestCase
from .wrap import TorchFxWrapTestCase
from .Node import TorchFxNodeTestCase
from .replace_pattern import TorchFxReplacepatternTestCase
from .symbolic_trace import TorchFxSymbolictraceTestCase
