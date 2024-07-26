import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.atan2)
class TorchTensorAtan2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atan2_correctness(self):
        """
        Test the correctness of torch.Tensor.atan2.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        result = tensor1.atan2(tensor2)
        return result
