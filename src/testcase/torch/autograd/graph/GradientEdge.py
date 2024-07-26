import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.graph.GradientEdge)
class TorchAutogradGraphGradientedgeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_gradient_edge_correctness(self):
        """
        Test the correctness of torch.autograd.graph.get_gradient_edge.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        a = torch.randn(input_size, requires_grad=True)  # tensor with requires_grad=True to enable gradient tracking
        b = torch.randn(input_size, requires_grad=True)  # tensor with requires_grad=True to enable gradient tracking
        c = a + b
        d = c.sum()
        d.backward()  # Calculate gradients
        gradient_edge = torch.autograd.grad(d, c)[0]  # Get gradient edge for tensor c
        return gradient_edge
