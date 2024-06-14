import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dsplit)
class TorchDsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dsplit_correctness(self):
    # Random dimensions for the tensor to ensure it has three or more dimensions
    dim1 = random.randint(1, 4)
    dim2 = random.randint(1, 4)
    dim3 = random.randint(1, 4)
    input_shape = [dim1, dim2, dim3]

    # Generate a random tensor with the specified dimensions
    tensor = torch.randn(input_shape)
    
    # Randomly choose whether indices_or_sections will be an integer or a list of integers
    if random.choice([True, False]):
        # If integer, ensure it evenly divides the third dimension
        while True:
            sections = random.randint(1, dim3)
            if dim3 % sections == 0:
                break
        result = torch.dsplit(tensor, sections)
    else:
        # If list, create a list of valid split indices in the range of the third dimension
        indices = sorted(random.sample(range(1, dim3), random.randint(1, dim3 - 1)))
        result = torch.dsplit(tensor, indices)
    
    return result
