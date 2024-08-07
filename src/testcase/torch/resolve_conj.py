import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.resolve_conj)
class TorchResolveUconjTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_resolve_conj_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.randn(input_size) + 1j * torch.randn(input_size)  # generate complex tensor
        input = input.conj()  # set conjugate bit to True
        result = torch.resolve_conj(input)
        return result
