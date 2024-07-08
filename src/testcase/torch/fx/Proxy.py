import torch
import torch.fx
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fx.Proxy)
class TorchFxProxyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_proxy_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Create a random tensor
        tensor = torch.randn(input_size)
    
        # Create a symbolic tracer
        class MyTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, module_qualified_name):
                return False
    
        tracer = MyTracer()
        graph = tracer.trace(torch.nn.Module())
        # Trace the tensor to create a Proxy object
        proxy = tracer.create_proxy('call_function', torch.relu, (tensor,), {})
    
        # Perform an operation using the Proxy object
        result = torch.add(proxy, proxy)
    
        return result
    