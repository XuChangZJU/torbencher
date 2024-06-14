import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.vsplit)
class TorchTensorVsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vsplit_correctness(self):
        # Randomly generate the number of rows and columns for the tensor
        num_rows = random.randint(2, 10)  # Ensure at least 2 rows for splitting
        num_cols = random.randint(1, 10)
        
        # Create a random tensor with the generated size
        tensor = torch.randn(num_rows, num_cols)
        
        # Randomly decide the number of splits or the sections
        if random.choice([True, False]):
            # Randomly choose a valid number of splits
            num_splits = random.randint(1, num_rows - 1)  # Ensure at least one split
            result = tensor.vsplit(num_splits)
        else:
            # Randomly generate valid sections
            sections = sorted(random.sample(range(1, num_rows), random.randint(1, num_rows - 1)))
            result = tensor.vsplit(sections)
        
        return result
    