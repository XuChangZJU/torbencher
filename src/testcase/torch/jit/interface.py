import random
from typing import List

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


class InterfaceType(torch.nn.Module):
    def run(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Impl1(InterfaceType):
    def run(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu()


class Impl2(InterfaceType):
    def __init__(self):
        super().__init__()
        self.val = torch.rand(())

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.val


@test_api(torch.jit.interface)
class TorchJitInterfaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def user_fn(impls: List[InterfaceType], idx: int, val: torch.Tensor) -> torch.Tensor:
        return impls[idx].run(val)

    def test_interface_correctness(self):
        # Randomly generate tensor dimensions and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        val = torch.randn(input_size)

        # Create instances of implementations
        impls = [Impl1(), Impl2()]

        # Randomly select an implementation index
        idx = random.randint(0, 1)

        # Call the JIT compiled function
        result = user_fn(impls, idx, val)
        return result
