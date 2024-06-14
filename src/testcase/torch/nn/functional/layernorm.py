import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.layernorm)
class TorchNnFunctionalLayernormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_layer_norm_correctness(self):
    # Random input size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Random input tensor
    input_tensor = torch.randn(input_size)

    # Random normalized_shape (last certain number of dimensions)
    normalized_shape = input_size[-random.randint(1, dim):]

    # Result of layer_norm
    result = torch.nn.functional.layer_norm(input_tensor, normalized_shape)
    return result
