import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.maximum)
class TorchTensorMaximumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maximum_correctness(self):
        """
        Test the correctness of torch.Tensor.maximum with small scale random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        tensor1 = torch.randn(input_size)  # Random tensor 1
        tensor2 = torch.randn(input_size)  # Random tensor 2 with the same size as tensor1
        result = tensor1.maximum(tensor2)  # Calculate the element-wise maximum of tensor1 and tensor2
        return result
    