import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.resize_)
class TorchTensorResizeUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_resize_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor with random size
        tensor = torch.randn(input_size)
        # Generate random new size, make sure the number of elements is different
        # new_size = input_size.copy()
        # while len(new_size) * new_size[0] == len(input_size) * input_size[0]:
        #     new_size = [random.randint(1, 5) for i in range(random.randint(1, 4))]
        # Generate random new size, ensure the number of elements is different
        while True:
            new_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
            if torch.tensor(new_size).numel() != tensor.numel():
                break

        # Resize the tensor
        tensor.resize_(new_size)
        return tensor
