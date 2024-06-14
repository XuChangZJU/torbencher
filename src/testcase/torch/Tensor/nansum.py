import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.nansum)
class TorchTensorNansumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nansum_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5)
        # Random input size
        input_size=[num_of_elements_each_dim for i in range(dim)]
        # Randomly generated tensor
        tensor = torch.randn(input_size)
        # Randomly replace some elements in tensor to NaN
        tensor[torch.rand(tensor.shape) < 0.5] = float('nan')
        # Calculate nansum
        result = tensor.nansum()
        return result
    