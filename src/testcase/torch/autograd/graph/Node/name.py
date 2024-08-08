import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.graph.Node.name)
class TorchAutogradGraphNodeNameTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_node_name_correctness(self):
        # Generate random tensor with requires_grad=True
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        a = torch.randn(input_size, requires_grad=True)

        # Perform an operation that creates a Node in the computation graph
        b = a.clone()

        # Get the Node object
        node = b.grad_fn

        # Get the name of the Node
        node_name = node.name()

        # Return the node name
        return node_name
