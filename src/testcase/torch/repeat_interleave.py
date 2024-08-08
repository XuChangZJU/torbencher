import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.repeat_interleave)
class TorchRepeatUinterleaveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_repeat_interleave_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generating random tensor
        input_tensor = torch.randn(input_size)
        # Generating random repeats tensor or integer
        if random.choice([True, False]):
            repeats = random.randint(1, 3)
        else:
            repeats = torch.randint(1, 4, (input_size[0],))

        # Randomly select a dimension
        dim_option = random.choice([None, random.randint(0, dim - 1)])
        # Applying repeat_interleave
        if dim_option is None:
            # 会将整个输入张量视为一维
            repeats = torch.randint(1, 4, (input_tensor.numel(),))
            result = torch.repeat_interleave(input_tensor, repeats)
        else:
            if isinstance(repeats, torch.Tensor) and repeats.size(0) != input_tensor.size(dim_option):
                repeats = torch.randint(1, 4, (input_tensor.size(dim_option),))
            result = torch.repeat_interleave(input_tensor, repeats, dim=dim_option)

        return result
