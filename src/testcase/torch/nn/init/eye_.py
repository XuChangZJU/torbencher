import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.nn.init.eye_)
class TorchNnInitEyeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_eye_correctness(self):
        # Randomly generate the dimension of the tensor
        dim = random.randint(1, 10)
        # Generate a random 2D tensor
        tensor = torch.randn(dim, dim)
        # Apply the eye_ function to the tensor
        result = torch.nn.init.eye_(tensor)
        return result
    