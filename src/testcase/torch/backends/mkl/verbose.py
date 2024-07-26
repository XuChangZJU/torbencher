import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mkl.verbose)
class TorchBackendsMklVerboseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_verbose_correctness(self):
        """
        Test the correctness of torch.backends.mkl.verbose.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        # Call torch.backends.mkl.verbose with VERBOSE_OFF
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_OFF):
            result_off = torch.add(tensor1, tensor2)

        # Call torch.backends.mkl.verbose with VERBOSE_ON
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
            result_on = torch.add(tensor1, tensor2)

        return result_off, result_on
