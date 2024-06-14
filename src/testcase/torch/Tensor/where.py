import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.where)
class TorchTensorWhereTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_where_correctness(self):
        # Randomly generate tensor dimensions and number of elements
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors and condition tensor
        tensor1 = torch.randn(input_size)  # tensor to select elements from
        tensor2 = torch.randn(input_size)  # tensor to select elements from
        condition_tensor = torch.randint(0, 2, input_size, dtype=torch.bool)  # Random boolean tensor for condition
    
        result = tensor1.where(condition_tensor, tensor2)
        return result
    