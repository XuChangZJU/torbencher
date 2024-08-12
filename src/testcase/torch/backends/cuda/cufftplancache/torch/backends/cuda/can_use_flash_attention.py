import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.can_use_flash_attention)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaCanUuseUflashUattentionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cufft_plan_cache_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch is compiled with CUDA support.")

        # Randomly generate the size of the cache
        cache_size = random.randint(1, 10)
        torch.backends.cuda.cufft_plan_cache.size = cache_size

        # Randomly generate the number of plans to add to the cache
        num_plans = random.randint(1, cache_size)

        # Add random plans to the cache
        for _ in range(num_plans):
            plan = torch.randn((random.randint(1, 4), random.randint(1, 4)), device='cuda')
            torch.backends.cuda.cufft_plan_cache.insert(plan)

        # Check the size of the cache
        result = torch.backends.cuda.cufft_plan_cache.size
        return result

    def test_can_use_flash_attention(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch is compiled with CUDA support.")

        # Check if flash attention can be used
        result = torch.backends.cuda.can_use_flash_attention()
        return result
