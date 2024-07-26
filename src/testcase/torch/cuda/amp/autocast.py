import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.amp.autocast)
class TorchCudaAmpAutocastTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_autocast_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")

        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Random tensor data
        tensor1 = torch.randn(input_size, device='cuda')
        tensor2 = torch.randn(input_size, device='cuda')

        # Using autocast to perform addition
        with torch.cuda.amp.autocast():
            result = torch.add(tensor1, tensor2)

        return result
