import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.LazyBatchNorm1d)
class TorchNnLazybatchnorm1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lazy_batch_norm_1d_correctness(self):
        # Random dimension for the tensor (batch size, num_features, length)
        batch_size = random.randint(1, 4)
        num_features = random.randint(1, 5)
        # Ensure length is at least 2 to avoid the error
        length = random.randint(2, 10)  # 修改这里，确保长度至少为2

        # Random input tensor
        input_tensor = torch.randn(batch_size, num_features, length)

        # Create LazyBatchNorm1d layer
        lazy_batch_norm = torch.nn.LazyBatchNorm1d()

        # Apply LazyBatchNorm1d to the input tensor
        result = lazy_batch_norm(input_tensor)

        return result
