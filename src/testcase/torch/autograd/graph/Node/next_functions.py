import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.graph.Node.next_functions)
class TorchAutogradGraphNodeNextUfunctionsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_node_next_functions_correctness(self):
        """
        Test the correctness of torch.autograd.graph.Node.next_functions.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        a = torch.randn(input_size, requires_grad=True)
        b = torch.randn(input_size, requires_grad=True)
        c = a + b
        d = c.mean()
        node = d.grad_fn.next_functions[0][0]
        next_functions = node.next_functions
        return next_functions
