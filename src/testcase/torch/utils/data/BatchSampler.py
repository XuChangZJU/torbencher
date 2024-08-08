import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.BatchSampler)
class TorchUtilsDataBatchsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_batchsampler_correctness(self):
        # Randomly generate sampler, batch_size, and drop_last
        sampler_length = random.randint(1, 10)
        sampler = torch.utils.data.SequentialSampler(range(sampler_length))
        batch_size = random.randint(1, sampler_length)
        drop_last = random.choice([True, False])

        # Create BatchSampler
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last)

        # Get the result
        result = list(batch_sampler)

        return result
