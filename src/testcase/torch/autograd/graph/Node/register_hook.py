import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.graph.Node.register_hook)
class TorchAutogradGraphNodeRegisterhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_node_register_hook_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        a = torch.randn(input_size, requires_grad=True)
        b = a.clone()
        handle = b.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
        b.sum().backward(retain_graph=True)
        result = a.grad
        handle.remove()
        return result
    