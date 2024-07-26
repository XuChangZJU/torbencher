import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.torch.utils.data.default_collate)
class TorchUtilsDataTorchUtilsDataDefaultcollateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_default_collate_correctness(self):
        # Generate random parameters for the input list
        list_length = random.randint(1, 10)  # Random length for the list
        # Create a list of tensors with random shapes and data types
        input_list = []
        for _ in range(list_length):
            dim = random.randint(1, 4)  # Random dimension for the tensors
            num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
            input_size = [num_of_elements_each_dim for i in range(dim)]
            tensor = torch.randn(input_size)
            input_list.append(tensor)
        result = torch.utils.data.dataloader.default_collate(input_list)
        return result
