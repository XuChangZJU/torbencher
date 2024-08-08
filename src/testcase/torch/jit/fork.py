import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.fork)
class TorchJitForkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_fork_correctness(self):
        # Define a simple function to be executed asynchronously
        def foo(a: torch.Tensor, b: int) -> torch.Tensor:
            return a + b

        # Generate random input tensor and integer
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        a = torch.randn(input_size)
        b = random.randint(1, 10)

        # Create a Future by forking the function execution
        fut = torch.jit.fork(foo, a, b)

        # Wait for the Future to complete and get the result
        result = torch.jit.wait(fut)
        return result
