import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.narrowcopy)
class TorchTensorNarrowcopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_narrow_copy_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension
        dimension = random.randint(0, len(input_size) - 1)
        # Randomly select a start index, making sure the operation is valid
        start = random.randint(0, input_size[dimension] - 1)
        # Randomly select a length, making sure the operation is valid
        length = random.randint(1, input_size[dimension] - start)
        # Perform narrow_copy operation
        result = input_tensor.narrow_copy(dimension, start, length)
        # Return the result tensor
        return result
    