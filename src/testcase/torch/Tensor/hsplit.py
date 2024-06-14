import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.hsplit)
class TorchTensorHsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hsplit_correctness(self):
        # Randomly choose the number of rows and columns for the tensor
        num_rows = random.randint(1, 10)
        num_cols = random.randint(2, 10)  # Ensure at least 2 columns for splitting
    
        # Generate a random 2D tensor with the chosen dimensions
        tensor = torch.randn(num_rows, num_cols)
    
        # Randomly choose the number of sections to split into or a split size
        if random.choice([True, False]):
            # Split by number of sections
            num_sections = random.randint(2, num_cols)  # Ensure at least 2 sections
            result = tensor.hsplit(num_sections)
        else:
            # Split by size of each section
            split_size = random.randint(1, num_cols - 1)  # Ensure valid split size
            result = tensor.hsplit(split_size)
    
        return result
    