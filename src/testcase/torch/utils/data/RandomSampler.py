import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.RandomSampler)
class TorchUtilsDataRandomsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip("RandomSampler not tested for now")
    def test_random_sampler_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        data_source = torch.randn(input_size)
        replacement = random.choice([True, False])  # Randomly choose replacement or not
        num_samples = random.randint(1, len(data_source)) if replacement else len(
            data_source)  # Randomly choose num_samples based on replacement
        sampler = torch.utils.data.RandomSampler(data_source, replacement, num_samples)
        result = list(sampler)
        return result
