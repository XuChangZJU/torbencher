import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.tile)
class TorchTensorTileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tile_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated tensor
        input_tensor = torch.randn(input_size)
        # Randomly generated dims (making sure the operation is valid by having the same length as the input tensor's dimension)
        dims = [random.randint(1, 5) for _ in range(dim)]
        # Perform the tile operation
        result = input_tensor.tile(dims)
        return result
    