import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.ldexp_)
class TorchTensorLdexpUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ldexp__correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor1
        tensor1 = torch.randn(input_size)
        # Generate random tensor2, make sure the shape of tensor1 and tensor2 are same
        tensor2 = torch.randn(input_size)
        # Call ldexp_ function 
        result = tensor1.ldexp_(tensor2)
        # Return the result so that we can check the effect of ldexp_
        return result
