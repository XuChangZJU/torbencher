import torch
import random
from torch.nn.utils.parametrize import is_parametrized

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.parametrize.is_parametrized)
class TorchNnUtilsParametrizeIsUparametrizedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_parametrized_with_parametrization(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(input_size[0], input_size[0])

        module = Model()
        tensor_name = 'linear.weight'
        result = is_parametrized(module, tensor_name)
        return result

    def test_is_parametrized_without_parametrization(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(input_size[0], input_size[0])

        module = Model()
        tensor_name = 'linear.bias'
        result = is_parametrized(module, tensor_name)
        return result
