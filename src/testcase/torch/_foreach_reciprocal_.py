import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_reciprocal_)
class TorchUforeachUreciprocalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_reciprocal_correctness(self):
        # foreach_reciprocal_ is an inplace function, so we test its correctness by comparing the result with torch.reciprocal
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.randn(input_size) for _ in range(random.randint(1, 3))]
        tensor_list_copy = [tensor.clone() for tensor in tensor_list]

        torch._foreach_reciprocal_(tensor_list)
        result = [torch.reciprocal(tensor) for tensor in tensor_list_copy]
        for i, (result_tensor, expected_tensor) in enumerate(zip(tensor_list, result)):
            assert torch.allclose(result_tensor, expected_tensor), f"Mismatch found in tensor {i}"
        return tensor_list
