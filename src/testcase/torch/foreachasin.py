import random
import torch


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.foreachasin)
class TorchForeachasinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_asin_correctness(self):
    num_tensors = random.randint(1, 4)  # Random number of tensors in the list
    tensor_list = []

    for _ in range(num_tensors):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)
        tensor_list.append(tensor)

    torch._foreach_asin_(tensor_list)
    return tensor_list
