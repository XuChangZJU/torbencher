import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.graph.Node.register_prehook)
class TorchAutogradGraphNodeRegisterUprehookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_node_register_prehook_correctness(self):
        # Randomly generate input tensor a
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        a = torch.randn(input_size, requires_grad=True)

        # Clone a to b
        b = a.clone()

        # Assert b.grad_fn is a Node
        assert isinstance(b.grad_fn, torch.autograd.graph.Node)

        # Define a pre-hook function that multiplies the gradient by 2
        def pre_hook(grad_outputs):
            return (grad_outputs[0] * 2,)

        # Register the pre-hook to b.grad_fn
        handle = b.grad_fn.register_prehook(pre_hook)

        # Compute the sum of b and backpropagate with retain_graph=True
        b.sum().backward(retain_graph=True)

        # Check if the gradient of a is doubled
        assert torch.allclose(a.grad, torch.ones_like(a) * 2)

        # Remove the pre-hook
        handle.remove()

        # Reset the gradient of a
        a.grad = None

        # Compute the sum of b and backpropagate again with retain_graph=True
        b.sum().backward(retain_graph=True)

        # Check if the gradient of a is back to normal
        assert torch.allclose(a.grad, torch.ones_like(a))

        return a.grad
