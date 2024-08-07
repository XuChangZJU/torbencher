import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.broadcast_tensors)
class TorchBroadcastUtensorsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_tensors_correctness(self):
        # Randomly generate dimensions for two tensors
        dim_tensor1 = random.randint(1, 3)  # Random dimension for tensor1 between 1 and 3
        dim_tensor2 = random.randint(1, 3)  # Random dimension for tensor2 between 1 and 3

        # Randomly generate the size of each dimension for both tensors, ensuring
        # that dimensions intended to be broadcasted have size 1 in at least one tensor.
        input_size_tensor1 = [random.randint(1, 5) for i in range(dim_tensor1)]
        input_size_tensor2 = [random.randint(1, 5) for i in range(dim_tensor2)]
        for i in range(min(dim_tensor1, dim_tensor2)):
            if input_size_tensor1[dim_tensor1 - i - 1] != input_size_tensor2[dim_tensor2 - i - 1]:
                if input_size_tensor1[dim_tensor1 - i - 1] == 1:
                    pass
                elif input_size_tensor2[dim_tensor2 - i - 1] == 1:
                    pass
                else:
                    input_size_tensor1[dim_tensor1 - i - 1] = 1

        # Generate the random tensors
        tensor1 = torch.randn(input_size_tensor1)
        tensor2 = torch.randn(input_size_tensor2)
        result = torch.broadcast_tensors(tensor1, tensor2)
        return result
