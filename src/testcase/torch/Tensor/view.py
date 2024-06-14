import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.view)
class TorchTensorViewTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_view_correctness(self):
    dim = random.randint(2, 4)  # Random dimension for the tensors, at least 2 dimensions
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for i in range(dim)]
    tensor = torch.randn(input_size)
    total_elements = torch.numel(tensor)
    new_shape = [total_elements]  # Reshape to a 1D tensor
    result = tensor.view(total_elements)
    return result
