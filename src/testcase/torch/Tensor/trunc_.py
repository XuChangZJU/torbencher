import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.trunc_)
class TorchTensorTruncUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_trunc__correctness(self):
        """
        Checks the correctness of torch.Tensor.trunc_() by comparing it with torch.trunc().
        The test generates random tensors and applies trunc_() in-place.
        It then compares the result with the output of torch.trunc() on the original tensor.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        expected_output = torch.trunc(input_tensor)
        input_tensor.trunc_()
        return input_tensor
