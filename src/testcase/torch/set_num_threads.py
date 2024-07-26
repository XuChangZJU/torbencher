import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.set_num_threads)
class TorchSetnumthreadsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_num_threads_correctness(self):
        # Random number of threads between 1 and 8
        num_threads = random.randint(1, 8)

        # Setting number of threads for intraop parallelism
        torch.set_num_threads(num_threads)

        # Creating large tensor to observe the effect of threading
        input_size = [random.randint(50, 100) for _ in range(3)]  # Random large tensor size
        tensor = torch.randn(input_size)

        def tensor_op(t):
            # A simple vectorized operation
            return torch.matmul(t, t.transpose(-2, -1))

        # Measuring time to ensure set_num_threads has visible effect
        import time
        start_time = time.time()
        result = tensor_op(tensor)
        end_time = time.time()
        execution_time = end_time - start_time

        return num_threads, execution_time
