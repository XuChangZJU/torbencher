import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.profile.self_cpu_time_total)
class TorchAutogradProfilerProfileSelfcputimetotalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_self_cpu_time_total_correctness(self):
        """
        Test if the self_cpu_time_total() method returns a float value,
        which represents the total CPU time spent on operations within the profiler's context.
        """
        with torch.autograd.profiler.profile(use_cuda=False) as prof:
            # Generate two random tensors
            dim = random.randint(1, 4)
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for i in range(dim)]
            tensor1 = torch.randn(input_size)
            tensor2 = torch.randn(input_size)
            # Perform some operations to measure CPU time
            result = torch.add(tensor1, tensor2)
            result = torch.mul(result, tensor1)
            result = torch.div(result, tensor2)
        total_cpu_time = prof.self_cpu_time_total
        return total_cpu_time
