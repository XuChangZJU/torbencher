import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.addcdiv)
class TorchTensorAddcdivTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_addcdiv_correctness(self):
        """
        Test the correctness of torch.Tensor.addcdiv.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors with the same size
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        tensor3 = torch.randn(input_size)

        # Calculate the expected result
        expected_result = tensor1 + (tensor2 / tensor3)

        # Calculate the result using torch.Tensor.addcdiv
        result = tensor1.addcdiv(tensor2, tensor3)

        return result
