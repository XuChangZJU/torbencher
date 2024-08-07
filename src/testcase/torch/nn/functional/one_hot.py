import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.one_hot)
class TorchNnFunctionalOneUhotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_one_hot_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random LongTensor
        input_tensor = torch.randint(0, 10, input_size).long()
        # Call one_hot
        result = torch.nn.functional.one_hot(input_tensor)
        return result
