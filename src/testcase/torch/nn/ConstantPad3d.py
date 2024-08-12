import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ConstantPad3d)
class TorchNnConstantpad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ConstantPad3d_correctness(self):
        # Random input size
        dim = random.randint(3, 5)  # Dimension should be at least 3 for 3D padding
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Random padding values
        padding_int = random.randint(1, 3)  # Random int for padding
        padding_tuple = (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3),
                         random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))  # Random tuple for padding

        # Random padding value
        padding_value = random.uniform(0.1, 10.0)

        # Test with int padding
        pad_int = torch.nn.ConstantPad3d(padding_int, padding_value)
        result_int_padding = pad_int(input_tensor)

        # Test with tuple padding
        pad_tuple = torch.nn.ConstantPad3d(padding_tuple, padding_value)
        result_tuple_padding = pad_tuple(input_tensor)

        # Return one result for simplicity, you can choose which one to return
        return result_int_padding
