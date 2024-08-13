import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.record_function)
class TorchAutogradProfilerRecordUfunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_record_function_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size, requires_grad=True)
        label = "label-" + str(random.randint(1, 100))  # Random label for the record function

        with torch.autograd.profiler.profile() as prof:
            y = tensor ** 2
            with torch.autograd.profiler.record_function(label):  # Label the block
                z = y ** 3
            z.sum().backward()  # Ensure the output is a scalar

        return prof.key_averages().table(sort_by="self_cpu_time_total")
