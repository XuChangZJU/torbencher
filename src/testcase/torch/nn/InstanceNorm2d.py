
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.InstanceNorm2d)
class TorchInstanceNorm2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instancenorm2d_correctness(self):
        num_features = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), num_features, random.randint(1, 10), random.randint(1, 10))
        instance_norm = torch.nn.InstanceNorm2d(num_features)
        result = instance_norm(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_instancenorm2d_large_scale(self):
        num_features = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), num_features, random.randint(100, 1000), random.randint(100, 1000))
        instance_norm = torch.nn.InstanceNorm2d(num_features)
        result = instance_norm(input_tensor)
        return result

