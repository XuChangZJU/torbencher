
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.ifftshift)
class TorchIfftshiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ifftshift_correctness(self):
        dim = random.randint(1, 10)  # Random dimension for the tensor
        tensor = torch.randn(dim, dtype=torch.complex64)
        result = torch.fft.ifftshift(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_ifftshift_large_scale(self):
        dim = random.randint(1000, 10000)  # Larger random dimension for the tensor
        tensor = torch.randn(dim, dtype=torch.complex64)
        result = torch.fft.ifftshift(tensor)
        return result


if __name__ == '__main__':
    run_tests()



