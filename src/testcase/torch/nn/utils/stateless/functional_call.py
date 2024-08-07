import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.stateless.functional_call)
class TorchNnUtilsStatelessFunctionalUcallTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_functional_call_correctness(self):
        # Define the size of the module parameters
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Define a simple module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(input_size))
                self.bias = torch.nn.Parameter(torch.randn(input_size))

            def forward(self, x):
                return x + self.weight + self.bias

        # Create an instance of the module
        module = MyModule()

        # Create random input
        x = torch.randn(input_size)

        # Create a dictionary of parameters and buffers
        parameters_and_buffers = {
            'weight': torch.randn(input_size),
            'bias': torch.randn(input_size),
        }

        # Call functional_call
        result = torch.nn.utils.stateless.functional_call(module, parameters_and_buffers, x)
        return result
