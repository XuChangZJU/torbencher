import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_set_to)
class TorchTensorIsUsetUtoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_set_to_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor1 = torch.randn(input_size)
        tensor2 = tensor1  # tensor2 is set to tensor1, they share the same memory
        result_true = tensor1.is_set_to(tensor2)

        tensor3 = torch.randn(input_size)  # tensor3 is a new tensor, does not share memory with tensor1
        result_false = tensor1.is_set_to(tensor3)

        return result_true, result_false
