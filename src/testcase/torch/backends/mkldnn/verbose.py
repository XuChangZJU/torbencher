import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mkldnn.verbose)
class TorchBackendsMkldnnVerboseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_verbose_correctness(self):
        # Randomly select verbose level
        verbose_level = random.choice([torch.backends.mkldnn.VERBOSE_OFF, torch.backends.mkldnn.VERBOSE_ON,
                                       torch.backends.mkldnn.VERBOSE_ON_CREATION])

        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        with torch.backends.mkldnn.verbose(verbose_level):
            result = torch.add(tensor1, tensor2)
        return result
