import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.not_equal)
class TorchNotequalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_not_equal_correctness(self):
        # Randomly generate the dimension of the input tensors
        dim = random.randint(1, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensors
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate two random tensors of the same size
        input1 = torch.randn(input_size)
        input2 = torch.randn(input_size)
        # Calculate the element-wise not equal comparison between the two tensors
        result = torch.not_equal(input1, input2)
        # Return the result tensor
        return result
