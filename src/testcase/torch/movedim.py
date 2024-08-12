import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


def shuffle(lst):
    return sorted(lst, key=lambda x: random.random())
@test_api(torch.movedim)
class TorchMovedimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_movedim_correctness(self):
        # Generate random dimensions and tensor size
        dim = random.randint(1, 5)
        length = random.randint(1, dim)
        size_of_tensor = [random.randint(1, 4) for _ in range(dim)]
        input_tensor = torch.randn(size_of_tensor)

        # Generate unique source and destination indices
        origin = list(range(dim)[:length])
        destination = list(range(dim)[:length])
        origin = shuffle(origin)
        destination = shuffle(destination)

        # Perform the movedim operation
        result = torch.movedim(input_tensor, origin, destination)

        # Return the result
        return result
