import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_erfc_)
class TorchUforeachUerfcUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_erfc_correctness(self):
        # foreach_erfc_ is an inplace function, so we test by comparing the result with torch.erfc
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.randn(input_size), torch.randn(input_size)]
        tensor_list_copy = [tensor.clone() for tensor in tensor_list]
        torch._foreach_erfc_(tensor_list)
        return [tensor1 - torch.erfc(tensor2) for tensor1, tensor2 in zip(tensor_list, tensor_list_copy)]
