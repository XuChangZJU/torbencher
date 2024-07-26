import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.matrix_exp)
class TorchLinalgMatrixexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_matrix_exp_correctness(self):
        # linalg.matrix_exp(A) -> Tensor
        # A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_size.extend([num_of_elements_each_dim, num_of_elements_each_dim])
        a = torch.randn(input_size)
        result = torch.linalg.matrix_exp(a)
        return result
