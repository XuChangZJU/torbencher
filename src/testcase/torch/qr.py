import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.qr)
class TorchQrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_qr_correctness(self):
    # Generate random dimension for the tensor
    dim = random.randint(1, 4)
    # Generate random number of elements each dimension, m >= n
    m = random.randint(1, 5)
    n = random.randint(1, m)
    input_size = [m, n]
    for i in range(dim - 1):
        input_size.insert(0, random.randint(1, 5))
    # Generate random tensor of shape input_size
    input_tensor = torch.randn(input_size)
    q, r = torch.qr(input_tensor)
    return q @ r
