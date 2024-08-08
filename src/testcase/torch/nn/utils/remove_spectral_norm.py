import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.remove_spectral_norm)
class TorchNnUtilsRemoveUspectralUnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_remove_spectral_norm_correctness(self):
        # Random dimension for the weight tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension for the weight tensor
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size for the weight tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random weight tensor
        weight = torch.randn(input_size)
        # Create a Linear layer and apply spectral norm
        linear_layer = torch.nn.Linear(in_features=input_size[0], out_features=input_size[0])
        linear_layer.weight = torch.nn.Parameter(weight)
        spectral_norm_linear_layer = torch.nn.utils.spectral_norm(linear_layer)
        # Remove spectral norm
        result = torch.nn.utils.remove_spectral_norm(spectral_norm_linear_layer)
        return result
