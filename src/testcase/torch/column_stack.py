import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.column_stack)
class TorchColumnUstackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_column_stack_correctness(self):
        # Define the number of tensors to stack
        num_tensors = random.randint(2, 5)

        # Generate random dimensions and sizes for tensors
        tensors = []
        for _ in range(num_tensors):
            # Randomly decide the dimension of each tensor (1D or 2D)
            dim = random.randint(1, 2)
            if dim == 1:
                # 1D tensor with random size
                size = random.randint(1, 5)
                tensor = torch.randn(size)
            else:
                # 2D tensor with random size
                size1 = random.randint(1, 5)
                size2 = random.randint(1, 5)
                tensor = torch.randn(size1, size2)
            tensors.append(tensor)

        # Ensure all tensors have the same size in dimension 0
        max_size0 = max(tensor.size(0) for tensor in tensors)
        for i in range(len(tensors)):
            if tensors[i].dim() == 1:
                tensors[i] = torch.cat([tensors[i], torch.zeros(max_size0 - tensors[i].size(0))])
            else:
                tensors[i] = torch.cat([tensors[i], torch.zeros(max_size0 - tensors[i].size(0), tensors[i].size(1))],
                                       dim=0)

        # Perform the column_stack operation
        result = torch.column_stack(tensors)
        return result
