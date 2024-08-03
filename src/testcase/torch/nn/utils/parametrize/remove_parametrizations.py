import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


class TestModule(nn.Module):
    def __init__(self, input_size):
        super(TestModule, self).__init__()
        self.param = nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        return x * self.param


@test_api(torch.nn.utils.parametrize.remove_parametrizations)
class TorchNnUtilsParametrizeRemoveparametrizationsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_remove_parametrizations_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        module = TestModule(input_size)
        tensor_name = 'param'

        # Apply a simple parametrization
        parametrize.register_parametrization(module, tensor_name, nn.Identity())

        # Remove the parametrization with leave_parametrized=True
        module_with_param = parametrize.remove_parametrizations(module, tensor_name, leave_parametrized=True)
        result_with_param = module_with_param.param

        # Apply the parametrization again
        parametrize.register_parametrization(module, tensor_name, nn.Identity())

        # Remove the parametrization with leave_parametrized=False
        module_without_param = parametrize.remove_parametrizations(module, tensor_name, leave_parametrized=False)
        result_without_param = module_without_param.param

        return result_with_param, result_without_param
