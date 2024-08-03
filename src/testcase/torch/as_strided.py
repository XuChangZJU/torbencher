import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.as_strided)
class TorchAsstridedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_strided_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = [random.randint(2, 5) for _ in
                                    range(dim)]  # Ensure each dimension has at least 2 elements

        tensor = torch.randn(num_of_elements_each_dim)

        view_size = tuple(random.randint(1, num_of_elements_each_dim[i]) for i in range(dim))  # Random view size
        view_stride = tuple(random.randint(1, num_of_elements_each_dim[i]) for i in range(dim))  # Random stride

        result = torch.as_strided(tensor, view_size, view_stride)
        return result
