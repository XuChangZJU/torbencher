import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.profile.key_averages)
class TorchAutogradProfilerProfileKeyaveragesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_key_averages_correctness(self):
        """
        Test the correctness of the key_averages function in the PyTorch profiler.
        """
        with torch.autograd.profiler.profile(record_shapes=True) as prof:
            # Generate random input tensors
            dim = random.randint(1, 4)
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for i in range(dim)]
            tensor1 = torch.randn(input_size)
            tensor2 = torch.randn(input_size)

            # Perform some operations
            result = torch.add(tensor1, tensor2)
            result = torch.mul(result, tensor1)
            result = torch.div(result, tensor2)

        # Test key_averages with different parameters
        avg_by_name = prof.key_averages()
        avg_by_input_shapes = prof.key_averages(True)
        avg_by_stack_n = prof.key_averages(False, random.randint(1, 5))  # group by top n stack trace entries

        return avg_by_name, avg_by_input_shapes, avg_by_stack_n
